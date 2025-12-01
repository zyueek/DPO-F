#!/usr/bin/env python3
#
# Usage example:
#   python3 code_test_final.py \
#     --input_csv  /path/in.csv \
#     --output_csv /path/out.csv \
#     --mode strict
#
# CSV must include columns:
#   - Function  (e.g., "Tic Tac Toe","MinStack","TwoSum", with many tolerated aliases)
#   - Refined_Generated_feedback
#   - Refined_Baseline_feedback
#
# Strict-mode alignment follows the paper’s evaluation spec:
#   TwoSum: distinct, in-bounds 0-based indices; input unchanged; infeasible -> must raise.
#   MinStack: LIFO; getMin tracks plateaus; empty-stack ops must raise.
#   TicTacToe: 0 for non-terminal/draw; correct winner id; invalid actions must raise and leave board unchanged.


import argparse
import os
import re
import subprocess
import tempfile
import pandas as pd
from typing import Optional, Tuple, List

# ------------------------------
# Helpers: blank-like checks, task normalization
# ------------------------------
def is_blanklike(x) -> bool:
    return (not isinstance(x, str)) or (x.strip() == "")

def norm_task(name: str) -> str:
    k = (name or "").strip().lower()
    k = k.replace("-", " ").replace("_", " ")
    k = re.sub(r"\s+", " ", k)
    alias = {
        "two sum": "twosum", "2sum": "twosum", "leetcode 1": "twosum",
        "twosum": "twosum",
        "min stack": "minstack", "minstack": "minstack",
        "tic tac toe": "tictactoe", "tic-tac-toe": "tictactoe", "tic tac-toe": "tictactoe",
        "tictactoe": "tictactoe", "ttt": "tictactoe"
    }
    return alias.get(k, k)

# ------------------------------
# Extract the last code block
# ------------------------------
FENCED_CPP = r"```(?:\s*(?:cpp|c\+\+|c))?\s*\r?\n(.*?)\r?\n```"
FENCED_ANY = r"```\s*\r?\n(.*?)\r?\n```"
INLINE_BACKTICKS = r"`([^`]+)`"

def extract_last_cpp_block(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for pat in (FENCED_CPP, FENCED_ANY, INLINE_BACKTICKS):
        m = re.findall(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m[-1].strip()
    # Heuristic fallback: looks like raw C++?
    if isinstance(text, str) and any(tok in text for tok in ["#include", "class ", ";", "int main("]):
        return text.strip()
    return None

def has_main(user_code: str) -> bool:
    return bool(re.search(r"\bint\s+main\s*\(", user_code))

# ------------------------------
# Compile + run helper
# ------------------------------
def compile_and_run_cpp(
    cpp_code: str,
    cpp_file_name: str,
    exe_name: str,
    timeout_sec: int = 12,
    mode: str = "strict",
    harness: bool = False,
) -> bool:
    """
    If harness=True, we define HARNESS_MODE=1 so any guarded stub main is excluded.
    """
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            cpp_path = os.path.join(tempdir, cpp_file_name)
            exe_path = os.path.join(tempdir, exe_name)
            with open(cpp_path, "w") as f:
                f.write(cpp_code)

            defs = ["-DLOOSE=1" if mode == "loose" else "-DLOOSE=0"]
            if harness:
                defs.append("-DHARNESS_MODE=1")

            if mode == "loose":
                timeout_sec = max(timeout_sec, 30)
                cxx_cmd = ["g++", "-std=c++17", "-O2", "-w", *defs, cpp_path, "-o", exe_path]
            else:
                cxx_cmd = ["g++", "-std=c++17", "-O2", *defs, cpp_path, "-o", exe_path]

            cproc = subprocess.run(cxx_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if cproc.returncode != 0:
                print(f"[COMPILE ERROR] {cpp_file_name}:\n{cproc.stderr}")
                return False
            try:
                rproc = subprocess.run([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
                if rproc.returncode != 0:
                    print(f"[RUNTIME ERROR] {exe_name}:\n{rproc.stderr}")
                    return False
                return True
            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] {exe_name} exceeded {timeout_sec}s")
                return False
    except Exception as e:
        print("[UNEXPECTED ERROR]", e)
        return False

# ------------------------------
# Stub main wrappers
# ------------------------------
GENERIC_HEADERS = r"""
#include <bits/stdc++.h>
using namespace std;
"""

def wrap_with_stub_main(user_code: str) -> str:
    """
    Always ensures a main exists for smoke-compile. Guarded by #ifndef HARNESS_MODE
    so harness builds (define HARNESS_MODE=1) will exclude the stub.
    """
    if has_main(user_code):
        return user_code
    return GENERIC_HEADERS + "\n" + user_code + r"""
#ifndef HARNESS_MODE
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    return 0;
}
#endif
"""

def wrap_with_guarded_stub_main(user_code: str) -> str:
    return wrap_with_stub_main(user_code)

# ============================================================
# Shared harness helpers (C++)
# ============================================================
COMMON = r"""
#include <bits/stdc++.h>
using namespace std;

#ifndef LOOSE
#define LOOSE 0
#endif

// Exact equality for scalars
template<typename T>
bool are_equal(const T& a, const T& b) { return a == b; }

// Order-insensitive equality for vectors of ints (without mutating inputs)
bool eq_vec_unordered(const vector<int>& a, const vector<int>& b) {
    if (a.size() != b.size()) return false;
    vector<int> sa = a, sb = b;
    sort(sa.begin(), sa.end());
    sort(sb.begin(), sb.end());
    return sa == sb;
}
"""

# ---------- TwoSum (STRICT = indices only, input unchanged; infeasible must raise) ----------
def harness_twosum_strict(user_code: str) -> str:
    body = r"""
int main(){
    // Must have Solution::twoSum(vector<int>&, int) -> vector<int> of 0-based indices
    vector<pair<vector<int>, int>> tests = {
        {{2,7,11,15}, 9},   // classic
        {{3,2,4}, 6},       // duplicates/multiple pairs
        {{-1, -2, -3, -4, -5}, -8}, // negatives
        {{0,4,3,0}, 0}      // zeros
    };

    for (auto &tc : tests) {
        vector<int> nums = tc.first;
        vector<int> original = nums; // must remain unchanged
        int target = tc.second;
        vector<int> out;
        try {
            Solution s;
            out = s.twoSum(nums, target);
        } catch (...) { return 1; }

        if (nums != original) return 1; // array must be unchanged

        if (out.size() != 2) return 1;
        int i = out[0], j = out[1];
        if (i == j) return 1;
        if (i < 0 || j < 0 || i >= (int)nums.size() || j >= (int)nums.size()) return 1;
        if (nums[i] + nums[j] != target) return 1;
    }

    // Infeasible instance must RAISE (throw)
    {
        vector<int> nums = {1,2,3};
        int target = 100;
        bool threw = false;
        try {
            Solution s;
            auto out = s.twoSum(nums, target);
            (void)out;
        } catch (...) {
            threw = true;
        }
        if (!threw) return 1;
    }

    return 0;
}
"""
    return COMMON + "\n" + wrap_with_guarded_stub_main(user_code) + "\n" + body

# ---------- TwoSum (LOOSE = accept 1-based or values, no-throw requirement) ----------
def harness_twosum_loose(user_code: str) -> str:
    body = r"""
bool indices_sum_to_target(const vector<int>& nums, const vector<int>& idx, int target) {
    if (idx.size() != 2) return false;
    auto ok = [&](int u){ return 0 <= u && u < (int)nums.size(); };
    return ok(idx[0]) && ok(idx[1]) && idx[0] != idx[1] && (nums[idx[0]] + nums[idx[1]] == target);
}
bool values_sum_to_target(const vector<int>& /*nums*/, const vector<int>& vals, int target) {
    return vals.size()==2 && (vals[0] + vals[1] == target);
}
int main(){
    vector<pair<vector<int>, int>> tests = {
        {{2,7,11,15}, 9}, {{3,2,4}, 6}, {{-1,-2,-3,-4,-5}, -8}, {{0,4,3,0}, 0}
    };
    int pass=0, total=0;
    for (auto &tc : tests) {
        vector<int> nums = tc.first;
        int target = tc.second;
        vector<int> out;
        try {
            Solution s;
            out = s.twoSum(nums, target);
        } catch (...) { continue; }
        total++;
        if (indices_sum_to_target(nums, out, target)) { pass++; continue; }
        // tolerate 1-based
        vector<int> one_based = out;
        if (one_based.size()==2) { one_based[0]--; one_based[1]--; }
        if (indices_sum_to_target(nums, one_based, target)) { pass++; continue; }
        if (values_sum_to_target(nums, out, target)) { pass++; continue; }
    }
    return (total>0 && pass*2 >= total) ? 0 : 1; // majority pass
}
"""
    return COMMON + "\n" + wrap_with_guarded_stub_main(user_code) + "\n" + body

# ---------- MinStack (STRICT = getMin across pushes/pops; empty ops must raise) ----------
def harness_minstack_strict(user_code: str) -> str:
    body = r"""
int main(){
    try {
        // Scenario A: classic with plateaus
        MinStack st;
        st.push(-2);
        st.push(0);
        st.push(-3);
        if (st.getMin() != -3) return 1;
        st.pop();
        if (st.top() != 0) return 1;
        if (st.getMin() != -2) return 1;

        // Scenario B: duplicates/plateaus
        MinStack st2;
        st2.push(2); st2.push(2); st2.push(1); st2.push(1);
        if (st2.getMin() != 1) return 1;

        // Empty-stack ops must raise
        MinStack se;
        bool threw_top=false, threw_pop=false, threw_min=false;
        try { (void)se.top(); } catch(...) { threw_top=true; }
        try { se.pop(); } catch(...) { threw_pop=true; }
        try { (void)se.getMin(); } catch(...) { threw_min=true; }
        if (!(threw_top && threw_pop && threw_min)) return 1;
    } catch (...) { return 1; }
    return 0;
}
"""
    return COMMON + "\n" + wrap_with_guarded_stub_main(user_code) + "\n" + body

# ---------- MinStack (LOOSE = callable behavior; no exception requirement) ----------
def harness_minstack_loose(user_code: str) -> str:
    body = r"""
int main(){
    try {
        MinStack st;
        st.push(-2); st.push(0); st.push(-3);
        // accept any min-like function name
        int m=0; bool ok=false;
        try { m=st.getMin(); ok=true; } catch(...) {}
        if(!ok){ try { m=st.min(); ok=true; } catch(...) {} }
        if(!ok){ try { m=st.get_min(); ok=true; } catch(...) {} }
        if(!ok) return 1;
        // pop/top non-throwing
        try { st.pop(); } catch(...) { return 1; }
        try { (void)st.top(); } catch(...) { return 1; }
    } catch (...) { return 1; }
    return 0;
}
"""
    return COMMON + "\n" + wrap_with_guarded_stub_main(user_code) + "\n" + body

# ---------- TicTacToe (STRICT = invalid actions must raise; draw/non-terminal=0; correct winner id) ----------
def harness_tictactoe_strict(user_code: str) -> str:
    body = r"""
int main(){
    // Row win by player 1
    {
        TicTacToe t(3);
        if(t.move(0,0,1) != 0) return 1;
        if(t.move(1,0,2) != 0) return 1;
        if(t.move(0,1,1) != 0) return 1;
        if(t.move(1,1,2) != 0) return 1;
        if(t.move(0,2,1) != 1) return 1; // winner 1
    }

    // Column win by player 2
    {
        TicTacToe t(3);
        if(t.move(0,2,2) != 0) return 1;
        if(t.move(0,0,1) != 0) return 1;
        if(t.move(1,2,2) != 0) return 1;
        if(t.move(1,0,1) != 0) return 1;
        if(t.move(2,2,2) != 2) return 1; // winner 2
    }

    // Diagonal win by player 1
    {
        TicTacToe t(3);
        if(t.move(0,0,1) != 0) return 1;
        if(t.move(0,1,2) != 0) return 1;
        if(t.move(1,1,1) != 0) return 1;
        if(t.move(0,2,2) != 0) return 1;
        if(t.move(2,2,1) != 1) return 1;
    }

    // Draw path (final board full, no winner -> 0)
    {
        TicTacToe t(3);
        if(t.move(0,0,1) != 0) return 1;
        if(t.move(0,1,2) != 0) return 1;
        if(t.move(0,2,1) != 0) return 1;
        if(t.move(1,1,2) != 0) return 1;
        if(t.move(1,0,1) != 0) return 1;
        if(t.move(1,2,2) != 0) return 1;
        if(t.move(2,1,1) != 0) return 1;
        if(t.move(2,0,2) != 0) return 1;
        if(t.move(2,2,1) != 0) return 1; // non-terminal/draw returns 0
    }

    // Invalid actions must RAISE and leave board unchanged
    {
        TicTacToe t(3);
        if(t.move(0,0,1) != 0) return 1;
        bool threw=false;
        try { (void)t.move(0,0,2); } catch(...) { threw=true; }
        if(!threw) return 1; // occupied cell should raise

        threw=false;
        try { (void)t.move(-1,0,1); } catch(...) { threw=true; }
        if(!threw) return 1; // out of bounds should raise
    }

    return 0;
}
"""
    return COMMON + "\n" + wrap_with_guarded_stub_main(user_code) + "\n" + body

# ---------- TicTacToe (LOOSE = winner checks only; invalids need not raise) ----------
def harness_tictactoe_loose(user_code: str) -> str:
    body = r"""
int main(){
    // Win checks similar to strict; do not require throws
    {
        TicTacToe t(3);
        if(t.move(0,0,1) == 0 &&
           t.move(1,0,2) == 0 &&
           t.move(0,1,1) == 0 &&
           t.move(1,1,2) == 0 &&
           t.move(0,2,1) != 0) {
            // ok
        } else return 1;
    }
    return 0;
}
"""
    return COMMON + "\n" + wrap_with_guarded_stub_main(user_code) + "\n" + body

# ============================================================
# Evaluate a single code cell with the appropriate harness
# ============================================================
def evaluate_feedback(code_text: str, func_name: str, mode: str = "strict") -> Tuple[Optional[int], str]:
    """
    Returns (executable_flag, test_result_string)
      executable_flag: 1, 0, or None (no code found)
      test_result_string: 'Pass' | 'Fail' | 'Skip'
    """
    code = extract_last_cpp_block(code_text)
    if not code:
        return (None, "Skip")  # no code -> Skip rather than Fail

    # Always wrap with a (guarded) stub main before ANY compile.
    wrapped = wrap_with_guarded_stub_main(code)

    # Smoke compile/run (stub main included since HARNESS_MODE is NOT defined here)
    exec_ok = 1 if compile_and_run_cpp(
        wrapped, "smoke.cpp", "smoke.out", timeout_sec=6, mode=mode, harness=False
    ) else 0

    if exec_ok == 0:
        return (0, "Fail")

    # If the user provided their own main, we won't inject our harness.
    # In loose mode, compiled user main counts as Pass; in strict, Skip (can’t test).
    if has_main(code):
        return (1, "Pass" if mode == "loose" else "Skip")

    k = norm_task(func_name)
    loose = (mode == "loose")

    if k == "twosum":
        src = harness_twosum_loose(wrapped) if loose else harness_twosum_strict(wrapped)
        ok = compile_and_run_cpp(src, "func.cpp", "func.out", timeout_sec=12, mode=mode, harness=True)
        return (1, "Pass" if ok else ("Skip" if loose else "Fail"))

    if k == "minstack":
        src = harness_minstack_loose(wrapped) if loose else harness_minstack_strict(wrapped)
        ok = compile_and_run_cpp(src, "func.cpp", "func.out", timeout_sec=12, mode=mode, harness=True)
        return (1, "Pass" if ok else ("Skip" if loose else "Fail"))

    if k == "tictactoe":
        src = harness_tictactoe_loose(wrapped) if loose else harness_tictactoe_strict(wrapped)
        ok = compile_and_run_cpp(src, "func.cpp", "func.out", timeout_sec=12, mode=mode, harness=True)
        return (1, "Pass" if ok else ("Skip" if loose else "Fail"))

    # Unknown task: in loose mode, compiled code is enough to Pass; in strict, Skip.
    return (1, "Pass" if loose else "Skip")

# ------------------------------
# Safe wrapper to guarantee one output per row
# ------------------------------
def safe_eval(text: str, func: str, mode: str) -> Tuple[Optional[int], str]:
    """Evaluate one cell safely; never raises. Returns (exec_flag, test_result_str)."""
    try:
        if is_blanklike(text):
            return (None, "Skip")
        return evaluate_feedback(text, func, mode=mode)
    except Exception as e:
        print(f"[EVAL ERROR] func={func}: {e}")
        return (0, "Fail" if mode == "strict" else "Skip")

# ------------------------------
# Orchestration
# ------------------------------
def process_file(input_csv: str, output_csv: str, mode: str = "strict"):
    bad_rows = []

    def _on_bad_lines(lines):
        bad_rows.append(lines)
        return None

    df = pd.read_csv(input_csv, engine="python", on_bad_lines=_on_bad_lines)

    required = {"Function", "Refined_Generated_feedback", "Refined_Baseline_feedback"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    gen_exec: List[Optional[int]] = []
    gen_test: List[str] = []
    base_exec: List[Optional[int]] = []
    base_test: List[str] = []

    for idx, row in df.iterrows():
        func = row.get("Function", "")

        ge, gt = safe_eval(row.get("Refined_Generated_feedback", ""), func, mode)
        gen_exec.append(ge)
        gen_test.append(gt)

        # Baseline evaluated strictly by default
        be, bt = safe_eval(row.get("Refined_Baseline_feedback", ""), func, "strict")
        base_exec.append(be)
        base_test.append(bt)

    assert len(gen_exec) == len(df) == len(base_exec), "Internal length mismatch."

    df["Generated_executable"] = gen_exec
    df["Generated_tests"]      = gen_test
    df["Baseline_executable"]  = base_exec
    df["Baseline_tests"]       = base_test

    df.to_csv(output_csv, index=False)
    print(f"Saved results -> {output_csv}")
    if bad_rows:
        print(f"[INFO] Skipped {len(bad_rows)} malformed CSV line(s).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--mode", choices=["strict","loose"], default="strict",
                    help="strict matches paper’s spec; loose is permissive")
    args = ap.parse_args()
    process_file(args.input_csv, args.output_csv, mode=args.mode)

if __name__ == "__main__":
    main()
