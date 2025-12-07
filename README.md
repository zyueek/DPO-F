# DPO-F+: Feedback-Aligned Code Repair

 

This repository is for the Artifact for CS 6367 – Final Project by Yueke Zhang, which contains data construction, model training (DPO + reward-augmented), inference, and evaluation pipelines for **feedback-aligned code repair**. As a PhD student, I intend to grow this into a full paper, so the implementation is relatively **complex and novel**. A full end-to-end reproduction of the training and evaluation pipeline is not feasible for a standard demo submission due to the following high-resource constraints:

- **Training.** We fine-tune small code models (e.g., 1.5B–7B) with DPO-style objectives on ~6k preference pairs. On 2× RTX A6000 GPUs, each run takes on the order of **2-4 GPU-hours**.

- **Dataset + labeling.** Building the preference data requires thousands of LLM-as-a-judge calls (for scoring feedback quality and correctness), which typically dominates **API cost** rather than GPU time.

- **Evaluation (SWE-bench Lite).** Running **SWE-bench Lite (300 real issues)** end-to-end, which involves Dockerized environments, dependency installs, and full test suites per issue and model, leading to **20-100** hours.

**So I provided demo on both training and inference of the LLM:**

**Training Demo:**

https://github.com/user-attachments/assets/2c22d3d6-1917-44ca-b0f4-f4117b1d8acc



**Inference Demo: LLM after DPO-F+ V.S Baseline:**

https://github.com/user-attachments/assets/b7dfc4a4-fabd-4952-8f8a-1c8bab457dac

It supports:

* Generating *natural-language feedback* for C++ fixes
* Running baseline vs DPO vs **DPO-f+** models
* Scoring feedback quality with a rubric (G-Eval style, 7 metrics)
* Pairwise A/B/Tie comparisons against a cross-family judge
* Optional code-compile smoke tests for corrected code

> If you’re skimming: jump to **[Quickstart](#quickstart)** and **[Typical Workflows](#typical-workflows)**.

---

## Repo structure

```
DPO-F/
├── data_construction/
│   ├── augmented_script.csv          # Seed/augmented items (code, metadata)
│   ├── feedback_alignment_label.py   # Build labels for alignment
│   ├── feedback_generation.py        # Generate feedback candidates (LLM)
│   ├── generated_code_test.py        # Try to compile/exe corrected C++ snippets
│   └── pair_generation.py            # Build preference pairs (chosen vs rejected)
├── evaluation/
│   ├── final_code_test.py            # Compile/exe test on final corrected code
│   ├── final_evaluation_score_only.py# Rubric scoring + pairwise A/B (no compile)
│   └── inference.py                  # Run model(s) to produce feedback outputs
└── training/
    └── train_dpo_reward.py           # Train DPO / DPO-f+ (LoRA, reward-aug)
```

> File names are descriptive; most scripts have a small config block at the top (paths, model IDs, etc.). Adjust as needed for your environment.

---

## Requirements

* Python 3.10+ (3.11 works too)
* CUDA available (optional but recommended)
* GCC (`g++`) for C++ compile tests

### Install

```bash
# From repo root
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Core libs (transformers/peft/torch; adjust CUDA wheel to your GPU/driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.42" peft accelerate sentencepiece
pip install pandas numpy scikit-learn tqdm
pip install openai==1.*        # OpenAI SDK (for rubric judge)
```

---

## API keys & environment

Some steps use LLM judges/generators. Export these (or set in your shell profile):

```bash
# Rubric judge (GPT-4 family)
export OPENAI_API_KEY="..."

# Pairwise judge (cross-family), if you use DeepSeek
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"   # default

# Optional: paths override for scripts that read envs
export INPUT_CSV="..."
export OUTPUT_CSV="..."
```

---

## Data expectations

At minimum, CSVs used across the pipeline contain:

* `Augmented_code`: original/buggy (or augmented) C++ snippet
* `Function`: (optional) target function name
* During/after inference you’ll have one or more of:

  * `Baseline_feedback`
  * `DPO_feedback` (optional, if you trained/ran DPO)
  * `Generated_feedback` (DPO-f+)

The generation/evaluation scripts will add columns with parsed code, compile logs, metric scores, and A/B results.

---

## Quickstart

1. **Generate feedback** (baseline / fine-tuned)
   Edit model names and input CSV paths at the top of `evaluation/inference.py`, then:

```bash
python -u evaluation/inference.py
```

2. **Score feedback (rubric) + pairwise A/B judge** (no compile):

```bash
python -u evaluation/final_evaluation_score_only.py
```

Outputs:

* `*_eval*.csv` with per-item metric scores (7 metrics + mean)
* `*_model_bootstrap_summary.csv` – model-level means + 95% CIs
* `*_model_bootstrap_diffs.csv` – paired diffs (DPO-f+ minus Baseline/DPO)
* `*_pairwise_tallies.csv` – win/loss/tie counts

3. **(Optional) Compile test corrected code**:

```bash
python -u evaluation/final_code_test.py
```

This tries to add a tiny `main()` wrapper and compile/run each corrected C++ candidate, logging success rate and errors.

---

## Typical workflows

### A. Build a preference dataset (for training)

1. **Seed/augment**: place or generate items in `data_construction/augmented_script.csv`.
2. **Generate candidate feedback**:

   ```bash
   python -u data_construction/feedback_generation.py
   ```
3. **Label / heuristics** (if you use weak labels):

   ```bash
   python -u data_construction/feedback_alignment_label.py
   ```
4. **Construct preference pairs** (chosen vs rejected):

   ```bash
   python -u data_construction/pair_generation.py
   ```

### B. Train DPO / DPO-f+

Configure LoRA, base model, and paths inside:

```bash
python -u training/train_dpo_reward.py
```

Artifacts (adapter weights) will be saved to your specified output dir.

### C. Inference (produce feedback)

With your chosen model(s) and adapters configured in `evaluation/inference.py`:

```bash
python -u evaluation/inference.py
```

This will create a CSV with feedback columns for each model you enabled.

### D. Evaluation (scores only)

Run rubric scoring + pairwise A/B:

```bash
python -u evaluation/final_evaluation_score_only.py
```

### E. Evaluation (compile smoke test)

```bash
python -u evaluation/final_code_test.py
```

---

## What the evaluators do

* **Rubric scoring (G-Eval style)**:
  Uses a GPT-4-family judge at `temperature=0.0`, 3 repeats.
  Metrics (1–5): *Conciseness, Quality, Explainability, Understandability, Completeness, Actionability, Contextual Relevance*.
  We report per-metric means and a simple average (“G-Eval mean”).

* **Pairwise A/B/Tie judge**:
  Cross-family judge (e.g., DeepSeek-V3), *deterministic* decoding.
  For each item we randomize A/B order and ask which feedback better helps the programmer fix the code.
  We tally **wins/losses/ties for DPO-f+** vs Baseline and (optionally) vs DPO.

* **Bootstrap summaries**:
  We include 95% CIs on model-level means and paired mean differences.

---

## Key script notes

* `evaluation/inference.py`
  Loads a base + (optionally) LoRA adapter, prompts in English, generates feedback, extracts code blocks from markdown fences, and can compile-check.

* `evaluation/final_evaluation_score_only.py`

* `evaluation/final_code_test.py`
  Minimal C++ compile/run test by injecting a tiny `main()`. Uses `g++` and timeouts; logs stderr on failure.

* `data_construction/*`
  Utilities for turning raw/augmented material into labeled pairs suitable for DPO/DPO-f+.

* `training/train_dpo_reward.py`
  DPO with a lightweight reward/margin signal (LoRA). Watch VRAM usage; gradient checkpointing is recommended for 7B.

---

## Reproducibility tips

* Set a global seed (most scripts honor `GLOBAL_SEED=12345`).
* Fix model versions (e.g., `codellama/CodeLlama-7b-Instruct`, `Qwen2.5-1.5B-Intruct`).
* Keep the evaluator prompts unchanged; mixing temperatures or families will affect scores.
* For long CSVs, adjust `batch_size` and add brief sleeps to respect API rate limits.

