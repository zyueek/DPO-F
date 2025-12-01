#!/usr/bin/env python3
"""
feedback_quality_eval.py
Usage:
  python feedback_quality_eval.py \
    --input  /path/in.csv \
    --output /path/out_scored.csv \
    --code-col Augmented_code \
    --feedback-col reasoning_steps \
    --model gpt-4o

"""

import os, sys, json, csv, argparse, random, time, re
import pandas as pd
from typing import Dict, Any, List, Tuple

# -----------------------
# OpenAI client (env-based)
# -----------------------
try:
    from openai import OpenAI
except Exception:
    print("Please install: pip install openai", file=sys.stderr); raise

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

# -----------------------
# Config
# -----------------------
METRICS = [
    "Conciseness",
    "Quality",
    "Explainability",
    "Understandability",
    "Completeness",
    "Actionability",
    "Contextual Relevance",
]

JUDGE_PROMPT = """You are evaluating NATURAL-LANGUAGE FEEDBACK that explains how to fix code.
You will be shown the original C++ code snippet and the feedback text. Rate the feedback ONLY
on the seven criteria below. Each score must be an integer from 1 to 5 (1=poor, 5=excellent).

Criteria (score each from 1–5, integers only):
- Conciseness: brief and to the point without unnecessary elaboration.
- Quality: technically correct guidance that addresses the actual error(s) and reflects sound practice.
- Explainability: clearly articulates the rationale behind changes (the "why").
- Understandability: easy to follow for an average programmer; avoids ambiguity.
- Completeness: covers all necessary changes to fully resolve the issue.
- Actionability: steps are directly applicable without extra interpretation.
- Contextual Relevance: provides sufficient context/conditions for applying the fix.

IMPORTANT:
- Judge ONLY the feedback text (not the corrected code’s exact compilation).
- Output STRICT JSON with exactly these keys, integers 1..5 only, no extra commentary.
  Example:
  {"Conciseness":5,"Quality":4,"Explainability":4,"Understandability":5,"Completeness":4,"Actionability":5,"Contextual Relevance":4}

Now evaluate the given item.

[CODE SNIPPET]
{code}

[FEEDBACK TEXT]
{feedback}

Return ONLY the JSON object, nothing else.
"""

JSON_RE = re.compile(r"\{[\s\S]*\}")

def call_judge(client: OpenAI, model: str, code: str, feedback: str, max_retries: int = 2) -> Dict[str, Any]:
    """Call the judge with temp=0 and enforce strict-JSON via retries."""
    prompt = JUDGE_PROMPT.format(code=str(code).strip(), feedback=str(feedback).strip())
    for attempt in range(max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # paper: temp 0.0
        )
        txt = (resp.choices[0].message.content or "").strip()
        # Try to locate JSON
        m = JSON_RE.search(txt)
        cand = m.group(0) if m else txt
        try:
            data = json.loads(cand)
            # validate keys & int values
            if sorted(data.keys()) != sorted(METRICS):
                raise ValueError("Wrong keys")
            for k,v in data.items():
                if not isinstance(v, int) or v < 1 or v > 5:
                    raise ValueError("Non-integer or out-of-range score")
            return data
        except Exception:
            # tighten the instruction and retry
            prompt = (JUDGE_PROMPT + "\nRemember: return ONLY a flat JSON object with the seven keys and integer values.")
            time.sleep(0.5)
    # If still malformed, return Nones to mark invalid
    return {k: None for k in METRICS}

def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def eval_item(client: OpenAI, model: str, code: str, feedback: str, n_repeats: int = 3) -> Tuple[Dict[str, float], float]:
    """Run n_repeats=3 independent judgments; average per metric; return (metric_means, g_eval)."""
    reps: List[Dict[str, Any]] = []
    for _ in range(n_repeats):
        reps.append(call_judge(client, model, code, feedback))
    per_metric_means: Dict[str, float] = {m: mean([r[m] for r in reps]) for m in METRICS}
    g_eval = mean(list(per_metric_means.values())) if all(per_metric_means[m] is not None for m in METRICS) else None
    return per_metric_means, g_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with at least the feedback column; code column recommended.")
    ap.add_argument("--output", required=True, help="Destination CSV with scores.")
    ap.add_argument("--code-col", default="Augmented_code", help="Column name for the original code snippet.")
    ap.add_argument("--feedback-col", default="reasoning_steps", help="Column name with feedback text to score.")
    ap.add_argument("--model", default="gpt-4o", help="Judge model (default: gpt-4o).")
    ap.add_argument("--repeats", type=int, default=3, help="Judging repeats per item (paper: 3).")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for row order.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on #rows for a quick run.")
    args = ap.parse_args()

    client = get_client()

    df = pd.read_csv(args.input)
    if args.feedback_col not in df.columns:
        print(f"ERROR: feedback column '{args.feedback_col}' not found.", file=sys.stderr); sys.exit(1)
    if args.code_col not in df.columns:
        print(f"WARNING: code column '{args.code_col}' not found; proceeding with empty code context.", file=sys.stderr)

    # Randomize item order to mitigate position effects (paper recommendation).
    # We still write results aligned to original order via index column.
    random.seed(args.seed)
    idxs = list(df.index)
    random.shuffle(idxs)
    if args.limit is not None:
        idxs = idxs[:args.limit]

    # Prepare output structure
    out_rows: List[Dict[str, Any]] = []
    for ridx in idxs:
        row = df.loc[ridx]
        code = row.get(args.code_col, "")
        feedback = row.get(args.feedback_col, "")
        if not isinstance(feedback, str) or not feedback.strip():
            # Skip empty feedbacks; still record row with Nones
            metric_means = {m: None for m in METRICS}
            g_eval = None
        else:
            metric_means, g_eval = eval_item(client, args.model, code, feedback, n_repeats=args.repeats)

        out = dict(row)
        for m in METRICS:
            out[f"{m}"] = metric_means[m]
        out["G_Eval"] = g_eval
        out_rows.append(out)

    # Reassemble in original index order for output CSV
    out_df = pd.DataFrame(out_rows).set_index(df.index.name or "index")
    out_df = out_df.reindex(df.index)  # sort back to original row order

    # Write
    # Keep original columns + the new metric columns + G_Eval
    base_cols = list(df.columns)
    score_cols = METRICS + ["G_Eval"]
    out_df = out_df[base_cols + score_cols]
    out_df.to_csv(args.output, index=False)
    print(f"Wrote scored CSV -> {args.output}")

if __name__ == "__main__":
    main()
