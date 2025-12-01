#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Input JSONL format (train/val/test):
  {
    "prompt": "...",                 # optional but recommended
    "chosen": "...",                 # preferred response text
    "rejected": "...",               # non-preferred response text
    "meta": {...}                    # optional
  }

"""

import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, PeftModel


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Stack simple tensors and keep prompt lengths
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            if vals[0].ndim == 0:
                out[k] = torch.tensor([v.item() for v in vals], dtype=torch.long)
            else:
                out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


# -------------------------
# Dataset
# -------------------------

class DPODataset(Dataset):
    """
    Expects a JSONL file with 'chosen', 'rejected', and (optionally) 'prompt'.
    If 'prompt' is present, the model will be trained with response-only loss masking.
    """
    def __init__(self, path: str, tokenizer, max_len: int = 1024,
                 subset_fraction: Optional[float] = None, subset_seed: int = 42):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        # Load JSON or JSONL
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                maybe = json.load(f)
                if isinstance(maybe, list):
                    self.data = maybe
                else:
                    raise ValueError("JSON must be a list or use JSONL format.")

        # Subset if requested
        if subset_fraction is not None and 0 < subset_fraction < 1:
            rng = random.Random(subset_seed)
            k = int(len(self.data) * subset_fraction)
            self.data = rng.sample(self.data, k)

        # Pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def _tok_concat(self, prompt_text: Optional[str], resp_text: str):
        """
        Return tokenized input_ids and attention_mask for (prompt + response)
        and compute prompt_len in tokens (so we can mask its loss later).
        """
        if prompt_text and prompt_text.strip():
            # Encode prompt and response separately, then concat
            p = self.tokenizer(prompt_text, add_special_tokens=False)
            r = self.tokenizer(resp_text, add_special_tokens=False)
            ids = p["input_ids"] + r["input_ids"]
            att = [1] * len(ids)
            # Truncate to max_len
            ids = ids[: self.max_len]
            att = att[: self.max_len]
            prompt_len = min(len(p["input_ids"]), len(ids))
        else:
            enc = self.tokenizer(resp_text, add_special_tokens=False)
            ids = enc["input_ids"][: self.max_len]
            att = [1] * len(ids)
            prompt_len = 0

        # Pad to max_len
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
            att = att + [0] * pad_len

        return torch.tensor(ids, dtype=torch.long), torch.tensor(att, dtype=torch.long), torch.tensor(prompt_len, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data[idx]
        prompt = row.get("prompt", None)
        chosen = row["chosen"]
        rejected = row["rejected"]

        chosen_ids, chosen_att, chosen_pL = self._tok_concat(prompt, chosen)
        rejected_ids, rejected_att, rejected_pL = self._tok_concat(prompt, rejected)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_att,
            "chosen_prompt_len": chosen_pL,  # for response-only masking
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_att,
            "rejected_prompt_len": rejected_pL,
        }


# -------------------------
# Model wrapper
# -------------------------

@dataclass
class LoRAArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # if None, we try common names


DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"]


class TrainerDPOFPlus:
    def __init__(
        self,
        base_model: str,
        output_dir: str,
        beta: float = 0.2,
        gamma: float = 0.05,
        lr: float = 2e-5,
        train_bs: int = 4,
        eval_bs: int = 8,
        seed: int = 42,
        bf16: bool = True,
        policy_lora: Optional[LoRAArgs] = None,
        reward_lora: Optional[LoRAArgs] = None,
        margin_lambda: float = 0.1,           # lightweight margin on reward margin (combined stage)
        margin_target: float = 1.0,           # target reward margin
    ):
        set_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.bf16 = bf16
        self.margin_lambda = margin_lambda
        self.margin_target = margin_target

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # LoRA configs
        self.policy_lora = policy_lora or LoRAArgs(r=16, alpha=32, dropout=0.05, target_modules=DEFAULT_TARGETS)
        self.reward_lora = reward_lora or LoRAArgs(r=32, alpha=64, dropout=0.05, target_modules=DEFAULT_TARGETS)

        # Models
        self.policy = None
        self.reference = None
        self.reward = None

    # ---- init/load/save

    def _load_base(self, trainable: bool, apply_lora: bool, lora_cfg: Optional[LoRAArgs]):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if trainable and apply_lora and lora_cfg:
            lcfg = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=lora_cfg.target_modules or DEFAULT_TARGETS,
                task_type="CAUSAL_LM",
                bias="none",
            )
            model = get_peft_model(model, lcfg)

        if not trainable:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        return model

    def save_lora(self, model, path):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # ---- dataloaders

    def build_loaders(self, train_path, val_path=None, test_path=None, max_len=1024,
                      train_subset=None, val_subset=None, test_subset=None):
        train_set = DPODataset(train_path, self.tokenizer, max_len, subset_fraction=train_subset)
        if val_path:
            val_set = DPODataset(val_path, self.tokenizer, max_len, subset_fraction=val_subset)
        else:
            # 90/10 split
            n = len(train_set)
            n_tr = int(0.9 * n)
            n_va = n - n_tr
            train_set, val_set = random_split(train_set, [n_tr, n_va], generator=torch.Generator().manual_seed(123))

        test_set = DPODataset(test_path, self.tokenizer, max_len, subset_fraction=test_subset) if test_path else None

        train_loader = DataLoader(train_set, batch_size=self.train_bs, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=self.eval_bs, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=self.eval_bs, shuffle=False, collate_fn=collate_fn) if test_set else None

        return train_loader, val_loader, test_loader

    # ---- sequence scoring (response-only masking aware)

    def _seq_logprob(self, model, input_ids, attention_mask, prompt_lens=None, length_normalize=False):
        """
        Returns sequence log-prob: sum_t log p(x_t | <t) over non-pad AND beyond prompt positions.
        If length_normalize = True, divide by effective length.
        """
        dev = next(model.parameters()).device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]  # shift
        labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:].float()

        # response-only masking: zero out positions that belong to the prompt
        if prompt_lens is not None:
            pL = prompt_lens.to(dev)  # shape: [B]
            # For each row i, mask first max(pL[i]-1, 0) label positions
            for i in range(input_ids.size(0)):
                k = max(int(pL[i].item()) - 1, 0)
                if k > 0:
                    mask[i, :k] = 0.0

        logp = F.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        tok_logp = tok_logp * mask
        seq_logp = tok_logp.sum(dim=-1)  # [B]

        if length_normalize:
            eff_len = mask.sum(dim=-1).clamp_min(1.0)
            seq_logp = seq_logp / eff_len
        return seq_logp

    def _kl_token_level(self, policy, reference, input_ids, attention_mask):
        dev = next(policy.parameters()).device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

        with torch.no_grad():
            ref_out = reference(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_out.logits[:, :-1, :]

        pol_out = policy(input_ids=input_ids, attention_mask=attention_mask)
        pol_logits = pol_out.logits[:, :-1, :]

        mask = attention_mask[:, 1:].float()
        pol_logp = F.log_softmax(pol_logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        pol_p = pol_logp.exp()

        kl_tok = (pol_p * (pol_logp - ref_logp)).sum(dim=-1)  # [B, T-1]
        kl_tok = kl_tok * mask
        kl = kl_tok.sum() / mask.sum().clamp_min(1.0)
        return kl

    # -------------------------
    # Stage 1: Policy DPO
    # -------------------------

    def train_policy(self, train_loader, val_loader, epochs=3):
        print("Initializing policy & reference...")
        self.policy = self._load_base(trainable=True, apply_lora=True, lora_cfg=self.policy_lora)
        self.reference = self._load_base(trainable=False, apply_lora=False, lora_cfg=None)

        opt = torch.optim.AdamW(self.policy.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

        best_val = float("inf")

        for ep in range(1, epochs + 1):
            self.policy.train()
            running = 0.0
            for step, batch in enumerate(train_loader, 1):
                opt.zero_grad()

                ch_ids = batch["chosen_input_ids"]
                ch_att = batch["chosen_attention_mask"]
                ch_pL  = batch["chosen_prompt_len"]

                rj_ids = batch["rejected_input_ids"]
                rj_att = batch["rejected_attention_mask"]
                rj_pL  = batch["rejected_prompt_len"]

                pol_ch = self._seq_logprob(self.policy, ch_ids, ch_att, ch_pL, length_normalize=False)
                pol_rj = self._seq_logprob(self.policy, rj_ids, rj_att, rj_pL, length_normalize=False)
                with torch.no_grad():
                    ref_ch = self._seq_logprob(self.reference, ch_ids, ch_att, ch_pL, length_normalize=False)
                    ref_rj = self._seq_logprob(self.reference, rj_ids, rj_att, rj_pL, length_normalize=False)

                # DPO with beta
                log_ratio = self.beta * ((pol_ch - pol_rj) - (ref_ch - ref_rj))
                dpo_loss = -F.logsigmoid(torch.clamp(log_ratio, -20, 20)).mean()

                # token-level KL
                kl = self._kl_token_level(self.policy, self.reference,
                                          torch.cat([ch_ids, rj_ids], 0),
                                          torch.cat([ch_att, rj_att], 0))

                loss = dpo_loss + self.gamma * kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                opt.step()
                sched.step()

                running += loss.item()
                if step % 50 == 0:
                    print(f"[Policy][Ep {ep}] Step {step}/{len(train_loader)}  Loss {loss.item():.4f}")

            val = self.eval_preference_accuracy(self.policy, val_loader)
            print(f"[Policy] Epoch {ep} done. TrainLoss {running/len(train_loader):.4f}  ValAcc {val['acc']:.4f}  ValLoss {val['loss']:.4f}")

            if val["loss"] < best_val:
                best_val = val["loss"]
                path = os.path.join(self.output_dir, "policy_best")
                self.save_lora(self.policy, path)
                print(f"Saved best policy to {path}")

        final_path = os.path.join(self.output_dir, "policy_final")
        self.save_lora(self.policy, final_path)
        print(f"Saved final policy to {final_path}")

    # -------------------------
    # Stage 2: Reward model
    # -------------------------

    def train_reward(self, train_loader, val_loader, epochs=3):
        print("Initializing reward model...")
        self.reward = self._load_base(trainable=True, apply_lora=True, lora_cfg=self.reward_lora)

        opt = torch.optim.AdamW(self.reward.parameters(), lr=self.lr, weight_decay=5e-4)
        total_steps = len(train_loader) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

        best_val = float("inf")

        for ep in range(1, epochs + 1):
            self.reward.train()
            run_loss = 0.0
            for step, batch in enumerate(train_loader, 1):
                opt.zero_grad()

                ch_ids = batch["chosen_input_ids"]
                ch_att = batch["chosen_attention_mask"]
                ch_pL  = batch["chosen_prompt_len"]
                rj_ids = batch["rejected_input_ids"]
                rj_att = batch["rejected_attention_mask"]
                rj_pL  = batch["rejected_prompt_len"]

                # length-normalized sequence rewards
                r_ch = self._seq_logprob(self.reward, ch_ids, ch_att, ch_pL, length_normalize=True)
                r_rj = self._seq_logprob(self.reward, rj_ids, rj_att, rj_pL, length_normalize=True)
                margin = r_ch - r_rj

                # Bradley–Terry / logistic preference loss
                bt_loss = -F.logsigmoid(margin).mean()

                loss = bt_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward.parameters(), 2.0)
                opt.step()
                sched.step()

                run_loss += loss.item()
                if step % 50 == 0:
                    acc = (margin > 0).float().mean().item()
                    print(f"[Reward][Ep {ep}] Step {step}/{len(train_loader)}  Loss {loss.item():.4f}  Acc {acc:.3f}")

            val = self.eval_preference_accuracy(self.reward, val_loader, length_normalize=True)
            print(f"[Reward] Epoch {ep} done. TrainLoss {run_loss/len(train_loader):.4f}  ValAcc {val['acc']:.4f}  ValLoss {val['loss']:.4f}")

            if val["loss"] < best_val:
                best_val = val["loss"]
                path = os.path.join(self.output_dir, "reward_best")
                self.save_lora(self.reward, path)
                print(f"Saved best reward to {path}")

        final_path = os.path.join(self.output_dir, "reward_final")
        self.save_lora(self.reward, final_path)
        print(f"Saved final reward to {final_path}")

    # -------------------------
    # Stage 3: Combined (DPO-f+)
    # -------------------------

    def train_combined(self, train_loader, val_loader, epochs=2, use_margin=True):
        assert self.policy is not None and self.reward is not None, "Train/load policy & reward first."
        if self.reference is None:
            self.reference = self._load_base(trainable=False, apply_lora=False, lora_cfg=None)

        # Ensure policy is trainable (LoRA adapters)
        for p in self.policy.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(self.policy.parameters(), lr=self.lr, weight_decay=5e-3, betas=(0.9, 0.95))
        total_steps = len(train_loader) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

        best_val = float("inf")

        for ep in range(1, epochs + 1):
            self.policy.train()
            run_loss = 0.0
            for step, batch in enumerate(train_loader, 1):
                opt.zero_grad()

                ch_ids = batch["chosen_input_ids"]
                ch_att = batch["chosen_attention_mask"]
                ch_pL  = batch["chosen_prompt_len"]
                rj_ids = batch["rejected_input_ids"]
                rj_att = batch["rejected_attention_mask"]
                rj_pL  = batch["rejected_prompt_len"]

                # policy seq log-probs
                pol_ch = self._seq_logprob(self.policy, ch_ids, ch_att, ch_pL, length_normalize=False)
                pol_rj = self._seq_logprob(self.policy, rj_ids, rj_att, rj_pL, length_normalize=False)

                # reference
                with torch.no_grad():
                    ref_ch = self._seq_logprob(self.reference, ch_ids, ch_att, ch_pL, length_normalize=False)
                    ref_rj = self._seq_logprob(self.reference, rj_ids, rj_att, rj_pL, length_normalize=False)

                # reward (length-normalized)
                with torch.no_grad():
                    rew_ch = self._seq_logprob(self.reward, ch_ids, ch_att, ch_pL, length_normalize=True)
                    rew_rj = self._seq_logprob(self.reward, rj_ids, rj_att, rj_pL, length_normalize=True)

                # Combine: policy + β * reward (paper's reward-augmented DPO)
                comb_ch = pol_ch + self.beta * rew_ch
                comb_rj = pol_rj + self.beta * rew_rj

                # DPO-style with β inside the total log-ratio for consistency
                log_ratio = self.beta * ((comb_ch - comb_rj) - (ref_ch - ref_rj))
                dpo_loss = -F.logsigmoid(torch.clamp(log_ratio, -20, 20)).mean()

                # token-level KL
                kl = self._kl_token_level(self.policy, self.reference,
                                          torch.cat([ch_ids, rj_ids], 0),
                                          torch.cat([ch_att, rj_att], 0))

                loss = dpo_loss + self.gamma * kl

                # optional lightweight margin on reward margin
                if use_margin and self.margin_lambda > 0:
                    reward_margin = (rew_ch - rew_rj)
                    margin_pen = F.relu(self.margin_target - reward_margin).mean()
                    loss = loss + self.margin_lambda * margin_pen

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                opt.step()
                sched.step()

                run_loss += loss.item()
                if step % 50 == 0:
                    print(f"[Combined][Ep {ep}] Step {step}/{len(train_loader)}  Loss {loss.item():.4f}")

            val = self.eval_combined_accuracy(val_loader)
            print(f"[Combined] Epoch {ep} done. TrainLoss {run_loss/len(train_loader):.4f}  "
                  f"ValAcc {val['acc']:.4f}  ValLoss {val['loss']:.4f}")

            if val["loss"] < best_val:
                best_val = val["loss"]
                path = os.path.join(self.output_dir, "combined_best")
                self.save_lora(self.policy, path)
                print(f"Saved best combined-policy to {path}")

        final_path = os.path.join(self.output_dir, "combined_final")
        self.save_lora(self.policy, final_path)
        print(f"Saved final combined-policy to {final_path}")

    # -------------------------
    # Evaluations
    # -------------------------

    @torch.no_grad()
    def eval_preference_accuracy(self, model, loader, length_normalize=False):
        model.eval()
        total = 0
        correct = 0
        losses = []

        for batch in loader:
            ch_ids = batch["chosen_input_ids"]
            ch_att = batch["chosen_attention_mask"]
            ch_pL  = batch["chosen_prompt_len"]
            rj_ids = batch["rejected_input_ids"]
            rj_att = batch["rejected_attention_mask"]
            rj_pL  = batch["rejected_prompt_len"]

            sc_ch = self._seq_logprob(model, ch_ids, ch_att, ch_pL, length_normalize=length_normalize)
            sc_rj = self._seq_logprob(model, rj_ids, rj_att, rj_pL, length_normalize=length_normalize)
            margin = sc_ch - sc_rj

            # logistic loss as diagnostic
            loss = -F.logsigmoid(margin).mean().item()
            losses.append(loss)

            total += ch_ids.size(0)
            correct += (margin > 0).sum().item()

        return {"acc": correct / max(total, 1), "loss": float(np.mean(losses)) if losses else 0.0}

    @torch.no_grad()
    def eval_combined_accuracy(self, loader):
        """
        Evaluate (policy + β*reward) under the DPO ratio vs reference.
        Report accuracy based on combined margins (chosen vs rejected).
        """
        assert self.policy is not None and self.reward is not None and self.reference is not None
        self.policy.eval(); self.reward.eval(); self.reference.eval()

        total = 0
        correct = 0
        losses = []

        for batch in loader:
            ch_ids = batch["chosen_input_ids"]
            ch_att = batch["chosen_attention_mask"]
            ch_pL  = batch["chosen_prompt_len"]
            rj_ids = batch["rejected_input_ids"]
            rj_att = batch["rejected_attention_mask"]
            rj_pL  = batch["rejected_prompt_len"]

            pol_ch = self._seq_logprob(self.policy, ch_ids, ch_att, ch_pL, length_normalize=False)
            pol_rj = self._seq_logprob(self.policy, rj_ids, rj_att, rj_pL, length_normalize=False)
            rew_ch = self._seq_logprob(self.reward, ch_ids, ch_att, ch_pL, length_normalize=True)
            rew_rj = self._seq_logprob(self.reward, rj_ids, rj_att, rj_pL, length_normalize=True)

            comb_ch = pol_ch + self.beta * rew_ch
            comb_rj = pol_rj + self.beta * rew_rj

            ref_ch = self._seq_logprob(self.reference, ch_ids, ch_att, ch_pL, length_normalize=False)
            ref_rj = self._seq_logprob(self.reference, rj_ids, rj_att, rj_pL, length_normalize=False)

            # same DPO diagnostic loss
            log_ratio = self.beta * ((comb_ch - comb_rj) - (ref_ch - ref_rj))
            loss = -F.logsigmoid(torch.clamp(log_ratio, -20, 20)).mean().item()
            losses.append(loss)

            # accuracy by margin
            margin = comb_ch - comb_rj
            total += ch_ids.size(0)
            correct += (margin > 0).sum().item()

        return {"acc": correct / max(total, 1), "loss": float(np.mean(losses)) if losses else 0.0}


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, type=str)
    ap.add_argument("--train", required=True, type=str, help="train.jsonl")
    ap.add_argument("--val", type=str, help="val.jsonl (optional)")
    ap.add_argument("--test", type=str, help="test.jsonl (optional)")
    ap.add_argument("--out", default="./dpo_fplus_out", type=str)

    ap.add_argument("--max_len", default=1024, type=int)
    ap.add_argument("--train_bs", default=4, type=int)
    ap.add_argument("--eval_bs", default=8, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--lr", default=2e-5, type=float)
    ap.add_argument("--beta", default=0.2, type=float, help="DPO β")
    ap.add_argument("--gamma", default=0.05, type=float, help="KL weight γ")
    ap.add_argument("--bf16", action="store_true")

    ap.add_argument("--policy_epochs", default=3, type=int)
    ap.add_argument("--reward_epochs", default=3, type=int)
    ap.add_argument("--combined_epochs", default=2, type=int)

    ap.add_argument("--train_subset", type=float)
    ap.add_argument("--val_subset", type=float)
    ap.add_argument("--test_subset", type=float)

    ap.add_argument("--margin_lambda", default=0.1, type=float)
    ap.add_argument("--margin_target", default=1.0, type=float)
    ap.add_argument("--no_margin", action="store_true")

    args = ap.parse_args()

    trainer = TrainerDPOFPlus(
        base_model=args.base_model,
        output_dir=args.out,
        beta=args.beta,
        gamma=args.gamma,
        lr=args.lr,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        seed=args.seed,
        bf16=args.bf16,
        policy_lora=LoRAArgs(r=16, alpha=32, dropout=0.05, target_modules=DEFAULT_TARGETS),
        reward_lora=LoRAArgs(r=32, alpha=64, dropout=0.05, target_modules=DEFAULT_TARGETS),
        margin_lambda=0.0 if args.no_margin else args.margin_lambda,
        margin_target=args.margin_target,
    )

    train_loader, val_loader, test_loader = trainer.build_loaders(
        args.train, args.val, args.test, max_len=args.max_len,
        train_subset=args.train_subset, val_subset=args.val_subset, test_subset=args.test_subset
    )

    # 1) Policy DPO
    trainer.train_policy(train_loader, val_loader, epochs=args.policy_epochs)

    # 2) Reward model
    trainer.train_reward(train_loader, val_loader, epochs=args.reward_epochs)

    # 3) Combined DPO-f+
    trainer.train_combined(train_loader, val_loader, epochs=args.combined_epochs, use_margin=not args.no_margin)

    # Final evaluations (optional)
    if test_loader is not None:
        pol_val = trainer.eval_preference_accuracy(trainer.policy, test_loader, length_normalize=False)
        rew_val = trainer.eval_preference_accuracy(trainer.reward, test_loader, length_normalize=True)
        cmb_val = trainer.eval_combined_accuracy(test_loader)
        print("\n[TEST] Policy  :", pol_val)
        print("[TEST] Reward  :", rew_val)
        print("[TEST] Combined:", cmb_val)


if __name__ == "__main__":
    main()
