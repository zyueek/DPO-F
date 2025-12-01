import json, os
def test_pairs_schema():
    p = "data/pairs_train.jsonl"
    if not os.path.exists(p): return
    with open(p) as f:
        line = f.readline().strip()
    if not line: return
    obj = json.loads(line)
    assert "prompt" in obj and "chosen" in obj and "rejected" in obj
