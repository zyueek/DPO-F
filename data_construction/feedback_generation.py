#!/usr/bin/env python3
"""
feedback_generation.py

Environment:
  OPENAI_API_KEY must be set.

Examples:
  python feedback_generation_updated.py prepare \
    --input /path/in.csv --output /path/out_prompts.csv --model gpt-4o

  python feedback_generation_updated.py infer \
    --input /path/in.csv --output /path/out_infer.csv --model gpt-4o
"""

import argparse
import csv
import os
import sys
from typing import Dict, List

import pandas as pd

# ---- Prompt templates copied verbatim from the paper ----
# Figure 3 — Prompt A/B/C (data preparation)  :contentReference[oaicite:2]{index=2}
PROMPT_A = (
    "You are a senior programming engineer and code reviewer.\n"
    "The [novice] programmer has written the following script:\n\n"
    "[script] .\n"
    "If the programmer’s code is well-written and functional, offer positive\n"
    "feedback; otherwise, provide clear, step-by-step guidance to help them\n"
    "identify the problem and then include corrected code derived from your\n"
    "guidance.\n"
    "Corrected Code: [insert corrected code here]"
)

PROMPT_B = (
    "You are a senior programming engineer and code reviewer.\n"
    "[Novice] programmers are required to complete a function\n\n"
    "[function_name] . The programmer has written the following script:\n\n"
    "[script] .\n"
    "If the programmer’s code is well-written and functional, offer positive\n"
    "feedback; otherwise, provide clear, step-by-step guidance to help them\n"
    "identify the problem and then include corrected code derived from your\n"
    "guidance.\n"
    "Corrected Code: [insert corrected code here]"
)

PROMPT_C = (
    "You are a senior programming engineer and code reviewer.\n"
    "[Novice] programmers are required to complete a function\n\n"
    "[function_name] . The task context is provided in [minstack.h]\n\n"
    "and [minstack.cpp] .\n\n"
    "The programmer has written the following script: [script] .\n"
    "If the programmer’s code is well-written and functional, offer positive\n"
    "feedback; otherwise, provide clear, step-by-step guidance to identify\n"
    "the problem and explain why it occurs, then include corrected code\n"
    "derived from your guidance.\n"
    "Corrected Code: [insert corrected code here]"
)

# Figure 4 — Inference Prompt (final evaluation)  :contentReference[oaicite:3]{index=3}
INFERENCE_PROMPT = (
    "The programmer has written the following script: [script] .\n"
    "Provide feedback to help correct and improve this code, then include\n"
    "the corrected code derived from your guidance."
)

# ---- Client (OpenAI-style) ----
try:
    from openai import OpenAI
except Exception as e:
    print("Please install openai: pip install openai", file=sys.stderr)
    raise

def get_client():
    # Requires OPENAI_API_KEY in environment; you can also set OPENAI_BASE_URL if needed.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL")  # optional
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def render(tpl: str, script: str, function_name: str = None) -> str:
    msg = tpl.replace("[script]", script.strip())
    if "[function_name]" in msg:
        msg = msg.replace("[function_name]", (function_name or "").strip())
    return msg

def call_model(client, model: str, prompt_text: str) -> str:
    # Paper prompts are single-message user prompts, no system role.
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0  # deterministic for reproducibility in data building/eval
    )
    return (resp.choices[0].message.content or "").strip()

def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize expected column name
    if "script" not in df.columns:
        if "Augmented_code" in df.columns:
            df = df.rename(columns={"Augmented_code": "script"})
        else:
            raise ValueError("Input CSV must contain a 'script' column (or 'Augmented_code').")
    return df

def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        print("No rows to write; produced empty output.", file=sys.stderr)
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def run_prepare(args):
    client = get_client()
    df = load_frame(args.input)
    out_rows = []
    for idx, row in df.iterrows():
        script = str(row["script"])
        function_name = str(row.get("Function", "") or "")

        # Prompt A
        pa = render(PROMPT_A, script)
        out_a = call_model(client, args.model, pa)

        # Prompt B
        pb = render(PROMPT_B, script, function_name=function_name)
        out_b = call_model(client, args.model, pb)

        # Prompt C
        pc = render(PROMPT_C, script, function_name=function_name)
        out_c = call_model(client, args.model, pc)

        base = dict(row)
        # Store the exact prompts & outputs for traceability
        base.update({
            "prompt_A_text": pa,
            "feedback_A": out_a,
            "prompt_B_text": pb,
            "feedback_B": out_b,
            "prompt_C_text": pc,
            "feedback_C": out_c,
        })
        out_rows.append(base)

        if (idx + 1) % 10 == 0:
            print(f"[prepare] processed {idx+1} rows...")

    save_csv(out_rows, args.output)
    print(f"[prepare] wrote: {args.output}")

def run_infer(args):
    client = get_client()
    df = load_frame(args.input)
    out_rows = []
    for idx, row in df.iterrows():
        script = str(row["script"])
        p = render(INFERENCE_PROMPT, script)
        out = call_model(client, args.model, p)

        base = dict(row)
        base.update({
            "inference_prompt_text": p,
            "feedback": out
        })
        out_rows.append(base)

        if (idx + 1) % 10 == 0:
            print(f"[infer] processed {idx+1} rows...")

    save_csv(out_rows, args.output)
    print(f"[infer] wrote: {args.output}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", required=True, help="Path to input CSV.")
    common.add_argument("--output", required=True, help="Path to output CSV.")
    common.add_argument("--model", required=True, help="Model name, e.g., gpt-4o-mini-2024-07-18")

    sp_prep = sub.add_parser("prepare", parents=[common], help="Generate Prompt A/B/C outputs")
    sp_inf  = sub.add_parser("infer", parents=[common], help="Generate Inference Prompt outputs")

    args = ap.parse_args()
    if args.mode == "prepare":
        run_prepare(args)
    elif args.mode == "infer":
        run_infer(args)
    else:
        ap.error("Unknown mode.")

if __name__ == "__main__":
    main()
