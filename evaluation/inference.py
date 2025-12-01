#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-accurate inference:
- Standardized Figure 4 prompt
- Identical decoding across models
- K=5 candidates at default temperature
- CodeLlama-7B-Instruct (or Qwen2.5-1.5B-Instruct) as base
"""

import argparse
import os
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

# Optional: PEFT adapters for DPO / DPO-f+
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

torch.set_grad_enabled(False)
torch.cuda.empty_cache()

# -----------------------------
# Utilities
# -----------------------------
FIGURE4_PROMPT_TEMPLATE = (
    "The programmer has written the following script:\n"
    "[script]\n"
    "Provide feedback to help correct and improve this code, then include "
    "the corrected code derived from your guidance."
)

def build_prompt_figure4(code: str) -> str:
    # exact structure used in the paper's Figure 4
    # (keep it minimal; do not add headings or extra sections)
    return FIGURE4_PROMPT_TEMPLATE.replace("[script]", f"```cpp\n{code}\n```")

def extract_code_block(text: str) -> str | None:
    """
    Extract the last fenced code block (``` ... ```). Returns None if not found.
    """
    try:
        matches = re.findall(r"```(?:cpp|c\+\+|c)?\n(.*?)\n```", text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
    except Exception:
        return None

# -----------------------------
# Inference (paper-accurate)
# -----------------------------
def load_model_and_tokenizer(base_model_name: str, adapter_path: str | None, device: str, dtype):
    """
    base_model_name: e.g., 'codellama/CodeLlama-7b-instruct-hf'
    adapter_path: None for baseline; path to PEFT adapter for DPO or DPO-f+
    """
    if adapter_path:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed but adapter_path was provided.")
        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        # if base_model_name is not specified, fall back to the adapter's base
        if base_model_name is None or base_model_name.strip() == "":
            base_model_name = peft_cfg.base_model_name_or_path

    print(f"[INFO] Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": device},
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval()

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"[INFO] Attaching PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).to(device).eval()

    return model, tokenizer

def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    device: str,
    use_autocast: bool,
    k: int = 5,
    max_new_tokens: int = 512,
):
    """
    Paper-accurate decoding:
      - do_sample=True (stochastic sampling)
      - default temperature (omit temperature arg)
      - no top_p / top_k / penalties
      - num_beams=1
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        # Avoid unnecessary padding that can shorten available space for generation
        padding=False,
        max_length=2048,   # safe upper bound for most 7B instruct context windows
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    generations = []
    for _ in range(k):
        with autocast(enabled=use_autocast):
            out = model.generate(
                **inputs,
                do_sample=True,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # IMPORTANT: do NOT set temperature/top_p/penalties here —
                # we want the model defaults (paper: "default temperature").
            )
        gen_tokens = out[0][input_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        generations.append(text)
    return generations

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Paper-accurate inference for feedback + corrected code.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV with at least columns: Augmented_code, Function (others are preserved).")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Where to save the CSV with generated outputs.")
    parser.add_argument("--base_model", type=str, default="codellama/CodeLlama-7b-instruct-hf",
                        help="Instruct base (paper used CodeLlama-7B-Instruct or Qwen2.5-1.5B-Instruct).")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to PEFT adapter (DPO or DPO-f+). Leave empty for baseline.")
    parser.add_argument("--k", type=int, default=5, help="Number of candidates per snippet (paper: K=5).")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    use_amp = (dtype == torch.float16)

    print(f"[INFO] Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    df = pd.read_csv(args.csv_path)
    if "Augmented_code" not in df.columns:
        raise ValueError("CSV must contain an 'Augmented_code' column.")
    if "Function" not in df.columns:
        # not strictly needed by the paper, but keep if present
        df["Function"] = ""

    # Drop rows without code, de-dup if needed (paper does this upstream)
    df = df.dropna(subset=["Augmented_code"]).reset_index(drop=True)

    # Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        device=device,
        dtype=dtype,
    )

    all_rows = []
    for i, row in df.iterrows():
        code = row["Augmented_code"]
        prompt = build_prompt_figure4(code)  # Figure 4, paper prompt

        gens = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            use_autocast=use_amp,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
        )

        for c_idx, text in enumerate(gens, start=1):
            all_rows.append({
                **row.to_dict(),
                "Candidate_ID": c_idx,
                "Prompt_Figure4": prompt,
                "Generated_feedback": text,
                "Extracted_code": (extract_code_block(text) or "No code block found"),
                "Base_model": args.base_model,
                "Adapter_path": args.adapter_path if args.adapter_path else "None",
                "Decoding": "do_sample=True, default temperature, num_beams=1"
            })

        if (i + 1) % 10 == 0:
            print(f"[INFO] Processed {i+1}/{len(df)} items...")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out_path, index=False)
    print(f"[DONE] Saved {len(out_df)} rows to: {args.out_path}")

if __name__ == "__main__":
    main()
