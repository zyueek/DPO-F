from evaluation.final_evaluation_score_only import coerce_full_scores
def test_coerce_full_scores():
    s = coerce_full_scores({"Conciseness":5,"Quality":4})
    assert s["Conciseness"]==5 and "Actionability" in s
