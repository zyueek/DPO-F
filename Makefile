PY=python

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

pairs:
\t$(PY) data_construction/pair_generation.py --input data_construction/augmented_script.csv --out data/pairs_train.jsonl

train:
\t$(PY) training/train_dpo_reward.py --config configs/train_codellama7b.yaml

infer:
\t$(PY) evaluation/inference.py --config configs/eval_defaults.yaml

score:
\t$(PY) evaluation/final_evaluation_score_only.py --config configs/eval_defaults.yaml

ctest:
\t$(PY) evaluation/final_code_test.py --input /home/zihan/dpo_programming/all_generated_feedback_codellma.csv

format:
\tpython -m pip install ruff && ruff check --fix .

test:
\tpytest -q

.PHONY: setup pairs train infer score ctest format test
