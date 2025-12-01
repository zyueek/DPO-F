#!/usr/bin/env python3
"""
pair_generation.py — Build accepted–rejected preference pairs for DPO/DPO-f+ training.


Inputs:
  A CSV with (some of) these columns (case-insensitive, flexible):
    group_id               : groups multiple candidates for the same snippet (RECOMMENDED)
    script | Augmented_code
    feedback | reasoning_steps | Refined_Generated_feedback
    revised_code | corrected_code
    exec_ok | Executable | Generated_executable | Baseline_executable (1/0 or True/False)
    tests   | test_result | Generated_tests | Baseline_tests         (expects 'Pass' when success)
    Conciseness, Quality, Explainability, Understandability, Completeness, Actionability, Contextual Relevance
    model   : (optional) which model produced the feedback
    prompt_id / prompt_type : (optional) A/B/C or 'infer'

Outputs:
  train.jsonl, val.jsonl, test.jsonl — each line:
    {
      "prompt": "...",                # standard Figure-4 inference prompt with [script] filled
      "chosen": "...",                # accepted feedback text (y+)
      "rejected": "...",              # rejected feedback text (y-)
      "meta": {
        "group_id": "...",
        "avg_score_chosen": 4.27,
        "avg_score_rejected": 3.11,
        "exec_pass_chosen": true,
        "exec_pass_rejected": false,
        "model_chosen": "gpt-4o",
        "model_rejected": "claude-3.5-sonnet",
        "prompt_type": "A"            # if available
      }
    }

Usage:
  python pair_generation.py --input /path/candidates.csv --outdir /path/pairs_out \
                       --seed 7 --train 0.85 --val 0.05 --test 0.10
"""

import argparse, os, sys, json, math, random
import pandas as pd
from typing import List, Dict, Any

INFER_PROMPT = (
    "The programmer has written the following script: {script} .\n"
    "Provide feedback to help correct and improve this code, then include "
    "the corrected code derived from your guidance."
)

# Column name aliases (case-insensitive)
ALIASES = {
    "group": ["group_id", "item_id", "sample_id", "pair_id", "code_id"],
    "script": ["script", "Augmented_code", "original_script", "code_snippet"],
    "feedback": ["feedback", "reasoning_steps", "Refined_Generated_feedback", "Generated_feedback", "feedback_text"],
    "revised": ["revised_code", "corrected_code", "Perfect_Script", "Refined_version", "code_revision"],
    "exec_flag": ["exec_ok", "Executable", "Generated_executable", "Baseline_executable", "exec_flag"],
    "tests": ["tests", "test_result", "Generated_tests", "Baseline_tests"],
    "model": ["model", "provider", "producer_model"],
    "prompt_type": ["prompt_id", "prompt_type"],
    # rubric metrics:
    "Conciseness": ["Conciseness"],
    "Quality": ["Quality"],
    "Explainability": ["Explainability"],
    "Understandability": ["Understandability"],
    "Completeness": ["Completeness"],
    "Actionability": ["Actionability"],
    "Contextual Relevance": ["Contextual Relevance", "Contextual_Relevance", "ContextualRelevance"],
}

RUBRIC_KEYS = [
    "Conciseness",
    "Quality",
    "Explainability",
    "Understandability",
    "Completeness",
    "Actionability",
    "Contextual Relevance",
]

def find_col(df: pd.DataFrame, names: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def coerce_bool(x) -> bool:
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("1","true","yes","y","pass","passed"): return True
        if t in ("0","false","no","n","fail","failed","error"): return False
    if isinstance(x, (int, float)):
        return bool(int(x))
    return False

def average_score(row: pd.Series) -> float:
    vals = []
    for k in RUBRIC_KEYS:
        v = row.get(k, None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = None
        if v is not None:
            vals.append(v)
    return sum(vals)/len(vals) if len(vals) == len(RUBRIC_KEYS) else float("nan")

def label_instance(exec_flag: bool, tests_ok: bool, avg_score: float) -> str:
    """
    Paper rule:
      accepted iff (exec & tests pass) AND (avg_score >= 4.0); else rejected.
    """
    if exec_flag and tests_ok and avg_score >= 4.0:
        return "accepted"
    return "rejected"

def build_pairs(df: pd.DataFrame, seed: int, split=(0.85,0.05,0.10), outdir="pairs_out"):
    os.makedirs(outdir, exist_ok=True)

    # Resolve columns
    col_group   = find_col(df, ALIASES["group"]) or "_implicit_group"
    col_script  = find_col(df, ALIASES["script"])
    col_fb      = find_col(df, ALIASES["feedback"])
    col_rev     = find_col(df, ALIASES["revised"])  # optional
    col_exec    = find_col(df, ALIASES["exec_flag"])
    col_tests   = find_col(df, ALIASES["tests"])
    col_model   = find_col(df, ALIASES["model"])
    col_ptype   = find_col(df, ALIASES["prompt_type"])

    # Basic checks
    missing = [n for n,v in {
        "script": col_script, "feedback": col_fb, "exec_flag": col_exec, "tests": col_tests
    }.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns (some alias acceptable): {missing}")

    # Ensure rubric columns exist
    for k in RUBRIC_KEYS:
        if k not in df.columns:
            raise ValueError(f"Missing rubric column '{k}' (exact name).")

    # If group id missing, synthesize one per-unique script row
    if col_group == "_implicit_group":
        df = df.copy()
        df[col_group] = df.groupby(col_script).ngroup()

    # Compute helper fields
    df["_avg_score"] = df.apply(average_score, axis=1)
    df["_exec_ok"]   = df[col_exec].apply(coerce_bool)
    df["_tests_ok"]  = df[col_tests].apply(lambda x: str(x).strip().lower() in ("pass","passed","0","ok","success"))
    df["_label"]     = df.apply(lambda r: label_instance(r["_exec_ok"], r["_tests_ok"], r["_avg_score"]), axis=1)

    # Group and make pairs
    pairs: List[Dict[str, Any]] = []
    for gid, g in df.groupby(col_group):
        script = str(g.iloc[0][col_script])
        accepted = g[g["_label"]=="accepted"]
        rejected = g[g["_label"]=="rejected"]
        if accepted.empty or rejected.empty:
            continue
        prompt_text = INFER_PROMPT.format(script=script.strip())
        for _, row_pos in accepted.iterrows():
            for _, row_neg in rejected.iterrows():
                item = {
                    "prompt": prompt_text,
                    "chosen": str(row_pos[col_fb]).strip(),
                    "rejected": str(row_neg[col_fb]).strip(),
                    "meta": {
                        "group_id": str(gid),
                        "avg_score_chosen": float(row_pos["_avg_score"]),
                        "avg_score_rejected": float(row_neg["_avg_score"]),
                        "exec_pass_chosen": bool(row_pos["_exec_ok"] and row_pos["_tests_ok"]),
                        "exec_pass_rejected": bool(row_neg["_exec_ok"] and row_neg["_tests_ok"]),
                        "model_chosen": str(row_pos.get(col_model, "")) if col_model else "",
                        "model_rejected": str(row_neg.get(col_model, "")) if col_model else "",
                        "prompt_type": str(row_pos.get(col_ptype, "")) if col_ptype else "",
                    }
                }
                pairs.append(item)

    # Shuffle + split 85/5/10
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    n_test  = n - n_train - n_val
    splits = {
        "train.jsonl": pairs[:n_train],
        "val.jsonl":   pairs[n_train:n_train+n_val],
        "test.jsonl":  pairs[n_train+n_val:],
    }

    for fname, items in splits.items():
        path = os.path.join(outdir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote {len(items):6d}  -> {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with scored & tested candidates")
    ap.add_argument("--outdir", default="pairs_out", help="Directory to write JSONL splits")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--train", type=float, default=0.85)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.10)
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")

    df = pd.read_csv(args.input)
    build_pairs(df, seed=args.seed, split=(args.train, args.val, args.test), outdir=args.outdir)

if __name__ == "__main__":
    main()
