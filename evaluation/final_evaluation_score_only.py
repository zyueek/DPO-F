#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feedback-alignment evaluation (paper-accurate)
- Rubric (G-Eval): GPT-4 family, temp=0.0, n=3 repeats → per-metric mean → overall mean.
- Pairwise (A/B/Tie): DeepSeek-V3, deterministic, randomized order per row, report win/loss/tie
  for Baseline vs DPO-f+ and DPO vs DPO-f+ separately.

Expected input CSV columns (strings with feedback text):
  - 'Augmented_code'          : source code snippet shown to judges
  - 'Baseline_feedback'       : baseline model feedback
  - 'DPO_feedback'            : standard DPO model feedback (optional)
  - 'Generated_feedback'      : DPO-f+ feedback (our model)

Outputs:
  - *_model_bootstrap_summary.csv, *_model_bootstrap_diffs.csv
  - Main CSV with per-item rubric scores and pairwise decisions
"""

import os
import re
import csv
import json
import math
import numpy as np
import pandas as pd
from time import sleep
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from openai import OpenAI

# =========================
# Config
# =========================
INPUT_CSV  = os.getenv("INPUT_CSV",  "all_generated_feedback_codellma.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "all_generated_feedback_codellma_eval_paper_correct.csv")
LOG_PATH   = os.getenv("LOG_PATH",   "test_part/g_eval_malformed.log")
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "12345"))

# Rubric judge (OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

RUBRIC_MODEL = os.getenv("RUBRIC_MODEL", "gpt-4o-mini")  # any GPT-4 family model is fine
RUBRIC_TEMPERATURE = 0.0
RUBRIC_REPEATS = 3

# Pairwise judge (DeepSeek-V3) — OpenAI-compatible client
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
if not DEEPSEEK_API_KEY:
    raise SystemExit("Set DEEPSEEK_API_KEY (pairwise judge)")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)
PAIRWISE_MODEL = os.getenv("PAIRWISE_MODEL", "deepseek-chat")  # DeepSeek-V3 chat
PAIRWISE_TEMPERATURE = 0.0  # deterministic

CRITERIA = [
    "Conciseness",
    "Quality",
    "Explainability",
    "Understandability",
    "Completeness",
    "Actionability",
    "Contextual Relevance",
]

# =========================
# Utilities
# =========================
rng_global = np.random.default_rng(GLOBAL_SEED)

def log_bad_generation(context: str, raw: str, err: str = ""):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[{datetime.now().isoformat()}] {context}\n")
            if err:
                f.write(f"Error: {err}\n")
            f.write((raw or "") + "\n")
    except Exception:
        pass  # logging must not break the run

def backoff_sleep(attempt: int, base: float = 1.0):
    sleep(base * (2 ** attempt))

def exponential_backoff_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit", "429", "temporarily", "overloaded"]):
                backoff_sleep(attempt, base=1.0)
            else:
                sleep(1.0)

# -------------------------
# Bootstrap helpers (95% CI)
# -------------------------
def _bootstrap_ci(values, n_boot=5000, alpha=0.05, agg=np.nanmean, random_state=0):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = agg(arr[idx], axis=1)
    point = float(agg(arr))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi

def _paired_bootstrap_diff_ci(a, b, n_boot=5000, alpha=0.05, agg=np.nanmean, random_state=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    diff = a - b
    return _bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, agg=agg, random_state=random_state)

# =========================
# Prompt builders & parsers
# =========================
def build_json_prompt(source_text: str, model_output: str) -> str:
    max_source_length = 2000
    max_output_length = 1200
    s = (source_text or "")
    o = (model_output or "")
    if len(s) > max_source_length:
        s = s[:max_source_length] + "... [truncated]"
    if len(o) > max_output_length:
        o = o[:max_output_length] + "... [truncated]"
    return f"""
You are an expert code reviewer. Evaluate the **AI-generated feedback** for the original code.

--- ORIGINAL SCRIPT ---
{s}

--- AI-GENERATED FEEDBACK ---
{o}

Return a pure JSON object with EXACTLY these 7 integer fields (1-5):
{{
  "Conciseness": <int>,
  "Quality": <int>,
  "Explainability": <int>,
  "Understandability": <int>,
  "Completeness": <int>,
  "Actionability": <int>,
  "Contextual Relevance": <int>
}}
Only output JSON.
"""

def extract_scores_fallback(text: str):
    scores = {}
    for crit in CRITERIA:
        patterns = [
            rf'"{crit}"\s*:\s*([1-5])',
            rf'{crit}\s*:\s*([1-5])',
            rf'{crit}[^0-9]([1-5])',
        ]
        val = None
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                try:
                    n = int(m.group(1))
                    if 1 <= n <= 5:
                        val = n
                        break
                except Exception:
                    pass
        scores[crit] = val if val is not None else np.nan
    return scores

def coerce_full_scores(d: dict) -> dict:
    out = {}
    for c in CRITERIA:
        v = d.get(c, np.nan)
        try:
            v = int(v)
        except Exception:
            v = np.nan
        if not (1 <= int(v) <= 5) if not np.isnan(v) else False:
            v = np.nan
        out[c] = v
    return out

# =========================
# Rubric G-eval (GPT-4 family)
# =========================
def rubric_eval(source: str, output: str, client: OpenAI, model: str, repeats: int) -> Dict[str, float]:
    prompt = build_json_prompt(source, output)
    samples = []
    for i in range(repeats):
        def api_call():
            return client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=RUBRIC_TEMPERATURE,
                max_tokens=120,
            )
        try:
            resp = exponential_backoff_retry(api_call)
            raw = (resp.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(raw)
                coerced = coerce_full_scores(parsed)
            except Exception as je:
                log_bad_generation("Rubric JSON parse failed; fallback", raw, str(je))
                coerced = coerce_full_scores(extract_scores_fallback(raw))
            samples.append(coerced)
        except Exception as e:
            log_bad_generation("Rubric API call failed", "(no content)", str(e))
            samples.append({c: np.nan for c in CRITERIA})
    avg = {c: float(np.nanmean([s[c] for s in samples])) for c in CRITERIA}
    return avg

# =========================
# Pairwise comparison (DeepSeek-V3)
# =========================
@dataclass
class PairwiseResult:
    decision: str  # "A", "B", "Tie"
    a_is: str      # label of A (e.g., "DPO-f+", "Baseline", "DPO")
    b_is: str      # label of B

PAIRWISE_INSTRUCTIONS = (
    "You are a strict software-engineering reviewer. "
    "Given the same code and two pieces of feedback (A and B), choose which one better explains "
    "to the programmer how to fix or improve the code. Consider clarity, technical correctness, "
    "explainability, completeness, actionability, and contextual relevance. "
    "Answer with exactly one token: 'A', 'B', or 'Tie'."
)

def pairwise_compare(code: str, text_A: str, text_B: str, client: OpenAI, model: str) -> str:
    prompt = f"""Code:
{code}

Feedback A:
{text_A}

Feedback B:
{text_B}

Answer with exactly one of: A, B, Tie.
"""
    def api_call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PAIRWISE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            temperature=PAIRWISE_TEMPERATURE,
            max_tokens=4,
        )
    resp = exponential_backoff_retry(api_call)
    ans = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\b(A|B|Tie)\b", ans, flags=re.IGNORECASE)
    return m.group(1).title() if m else "Tie"

def compare_two_models(code: str, name1: str, txt1: Optional[str], name2: str, txt2: Optional[str]) -> Optional[PairwiseResult]:
    if not (isinstance(txt1, str) and txt1.strip()) or not (isinstance(txt2, str) and txt2.strip()):
        return None
    # Randomize A/B to mitigate position bias
    if rng_global.random() < 0.5:
        A_text, A_name, B_text, B_name = txt1, name1, txt2, name2
    else:
        A_text, A_name, B_text, B_name = txt2, name2, txt1, name1
    decision = pairwise_compare(code, A_text, B_text, deepseek_client, PAIRWISE_MODEL)
    return PairwiseResult(decision=decision, a_is=A_name, b_is=B_name)

# =========================
# Full pipeline
# =========================
def run_full_pipeline(input_csv: str, output_csv: str, batch_size: int = 10):
    df = pd.read_csv(
        input_csv,
        on_bad_lines="skip",
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )

    # Sanity: expected columns
    expected = ["Augmented_code", "Generated_feedback", "Baseline_feedback"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. "
                         f"Expected at least {expected} (DPO_feedback optional).")

    has_dpo = "DPO_feedback" in df.columns

    print(f"\n=== STEP 1: Rubric scoring (GPT-4 family, temp=0.0, repeats=3) on {len(df)} rows ===")
    rub_gen_rows, rub_base_rows, rub_dpo_rows = [], [], []

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        print(f" Scoring rows {batch_start+1}-{batch_end} …")
        for idx in range(batch_start, batch_end):
            row = df.iloc[idx]
            code = str(row.get("Augmented_code", "") or "")

            # DPO-f+ (Generated_feedback)
            gen_text = row.get("Generated_feedback", "")
            gen_scores = rubric_eval(code, str(gen_text or ""), openai_client, RUBRIC_MODEL, RUBRIC_REPEATS) \
                        if isinstance(gen_text, str) and gen_text.strip() else {c: np.nan for c in CRITERIA}
            gen_scores["G_eval_avg_score"] = float(np.nanmean(list(gen_scores.values())))
            rub_gen_rows.append(gen_scores)

            # Baseline
            base_text = row.get("Baseline_feedback", "")
            base_scores = rubric_eval(code, str(base_text or ""), openai_client, RUBRIC_MODEL, RUBRIC_REPEATS) \
                        if isinstance(base_text, str) and base_text.strip() else {c: np.nan for c in CRITERIA}
            base_scores["G_eval_avg_score"] = float(np.nanmean(list(base_scores.values())))
            rub_base_rows.append(base_scores)

            # DPO (optional)
            if has_dpo:
                dpo_text = row.get("DPO_feedback", "")
                dpo_scores = rubric_eval(code, str(dpo_text or ""), openai_client, RUBRIC_MODEL, RUBRIC_REPEATS) \
                            if isinstance(dpo_text, str) and dpo_text.strip() else {c: np.nan for c in CRITERIA}
                dpo_scores["G_eval_avg_score"] = float(np.nanmean(list(dpo_scores.values())))
                rub_dpo_rows.append(dpo_scores)

            sleep(0.05)

    df = pd.concat([df,
                    pd.DataFrame(rub_gen_rows).add_prefix("Generated_"),
                    pd.DataFrame(rub_base_rows).add_prefix("Baseline_")],
                   axis=1)
    if has_dpo:
        df = pd.concat([df, pd.DataFrame(rub_dpo_rows).add_prefix("DPO_")], axis=1)

    # =========================
    # STEP 2: Pairwise comparisons (DeepSeek-V3)
    # =========================
    print("\n=== STEP 2: Pairwise comparisons (DeepSeek-V3, deterministic, randomized A/B) ===")
    # Baseline vs DPO-f+
    results_b_vs_gen = []
    # DPO vs DPO-f+ (if available)
    results_dpo_vs_gen = []

    for idx, row in df.iterrows():
        code = str(row.get("Augmented_code", "") or "")
        gen = row.get("Generated_feedback", "")
        base = row.get("Baseline_feedback", "")

        res1 = compare_two_models(code, "Baseline", base, "DPO-f+", gen)
        results_b_vs_gen.append(res1)

        if has_dpo:
            dpo = row.get("DPO_feedback", "")
            res2 = compare_two_models(code, "DPO", dpo, "DPO-f+", gen)
            results_dpo_vs_gen.append(res2)

        if idx % 25 == 0:
            sleep(0.2)

    def unpack_results(prefix: str, results):
        cols = {
            f"{prefix}_decision": [],
            f"{prefix}_A_is": [],
            f"{prefix}_B_is": [],
        }
        for r in results:
            if r is None:
                cols[f"{prefix}_decision"].append(np.nan)
                cols[f"{prefix}_A_is"].append(np.nan)
                cols[f"{prefix}_B_is"].append(np.nan)
            else:
                cols[f"{prefix}_decision"].append(r.decision)
                cols[f"{prefix}_A_is"].append(r.a_is)
                cols[f"{prefix}_B_is"].append(r.b_is)
        return pd.DataFrame(cols)

    df = pd.concat([df, unpack_results("pairwise_baseline_vs_gen", results_b_vs_gen)], axis=1)
    if has_dpo:
        df = pd.concat([df, unpack_results("pairwise_dpo_vs_gen", results_dpo_vs_gen)], axis=1)

    # =========================
    # STEP 3: Coerce numeric & summaries
    # =========================
    for col in df.columns:
        if any(col.endswith(c) for c in CRITERIA) or col.endswith("G_eval_avg_score"):
            with np.errstate(all="ignore"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bootstrap model-level summaries and paired diffs (Generated vs Baseline, and vs DPO)
    print("\n=== STEP 3: Bootstrap summaries (95% CI) ===")
    summary_rows = []
    def add_metric(prefix: str, label: str, metric: str):
        col = f"{prefix}_{metric}"
        if col in df.columns:
            mean_, lo, hi = _bootstrap_ci(df[col].values, n_boot=5000, alpha=0.05, agg=np.nanmean, random_state=GLOBAL_SEED)
            summary_rows.append({
                "Model": label,
                "Metric": metric,
                "Mean": round(mean_, 4) if not math.isnan(mean_) else np.nan,
                "CI_lower": round(lo, 4) if not math.isnan(lo) else np.nan,
                "CI_upper": round(hi, 4) if not math.isnan(hi) else np.nan,
                "N_non_nan": int(np.isfinite(df[col].values).sum())
            })

    for m in CRITERIA + ["G_eval_avg_score"]:
        add_metric("Generated", "DPO-f+", m)
        add_metric("Baseline", "Baseline", m)
        if has_dpo:
            add_metric("DPO", "DPO", m)

    diffs_rows = []
    def add_diff(m: str, col_a: str, col_b: str, name: str):
        if col_a in df.columns and col_b in df.columns:
            mean_diff, lo, hi = _paired_bootstrap_diff_ci(
                df[col_a].values, df[col_b].values,
                n_boot=5000, alpha=0.05, agg=np.nanmean, random_state=GLOBAL_SEED+1
            )
            n_pairs = int((~df[col_a].isna() & ~df[col_b].isna()).sum())
            diffs_rows.append({
                "Metric": m,
                "Comparison": name,
                "MeanDiff": round(mean_diff, 4) if not math.isnan(mean_diff) else np.nan,
                "CI_lower": round(lo, 4) if not math.isnan(lo) else np.nan,
                "CI_upper": round(hi, 4) if not math.isnan(hi) else np.nan,
                "N_pairs": n_pairs
            })

    for m in CRITERIA + ["G_eval_avg_score"]:
        add_diff(m, f"Generated_{m}", f"Baseline_{m}", "DPO-f+ minus Baseline")
        if has_dpo:
            add_diff(m, f"Generated_{m}", f"DPO_{m}", "DPO-f+ minus DPO")

    summary_df = pd.DataFrame(summary_rows)
    diffs_df = pd.DataFrame(diffs_rows)

    _out_base = os.path.splitext(OUTPUT_CSV)[0]
    summary_path = _out_base + "_model_bootstrap_summary.csv"
    diffs_path   = _out_base + "_model_bootstrap_diffs.csv"
    summary_df.to_csv(summary_path, index=False)
    diffs_df.to_csv(diffs_path, index=False)
    print(f"Saved bootstrap summaries:\n  {summary_path}\n  {diffs_path}")

    # =========================
    # STEP 4: Pairwise tallies (win/loss/tie)
    # =========================
    def tally_pairwise(col_decision, col_Ais, col_Bis, target_vs="Baseline vs DPO-f+"):
        sub = df[[col_decision, col_Ais, col_Bis]].dropna()
        wins = losses = ties = 0
        for _, r in sub.iterrows():
            dec = str(r[col_decision]).strip()
            Ais = str(r[col_Ais]).strip()
            Bis = str(r[col_Bis]).strip()
            # We care about DPO-f+ vs the other model in this comparison
            if dec == "Tie":
                ties += 1
            elif dec == "A":
                # A wins
                if Ais == "DPO-f+":
                    wins += 1
                else:
                    losses += 1
            elif dec == "B":
                if Bis == "DPO-f+":
                    wins += 1
                else:
                    losses += 1
        return pd.DataFrame([{
            "Comparison": target_vs,
            "Wins_for_DPO-f+": wins,
            "Losses_for_DPO-f+": losses,
            "Ties": ties,
            "Total_compared": (wins + losses + ties)
        }])

    tallies = []
    tallies.append(tally_pairwise("pairwise_baseline_vs_gen_decision",
                                  "pairwise_baseline_vs_gen_A_is",
                                  "pairwise_baseline_vs_gen_B_is",
                                  "Baseline vs DPO-f+"))
    if has_dpo:
        tallies.append(tally_pairwise("pairwise_dpo_vs_gen_decision",
                                      "pairwise_dpo_vs_gen_A_is",
                                      "pairwise_dpo_vs_gen_B_is",
                                      "DPO vs DPO-f+"))

    tallies_df = pd.concat(tallies, ignore_index=True)
    tallies_path = _out_base + "_pairwise_tallies.csv"
    tallies_df.to_csv(tallies_path, index=False)
    print(f"Saved pairwise tallies:\n  {tallies_path}")

    # =========================
    # Save main results
    # =========================
    df.to_csv(output_csv, index=False)

# =========================
# Main
# =========================
if __name__ == "__main__":
    run_full_pipeline(INPUT_CSV, OUTPUT_CSV, batch_size=10)
