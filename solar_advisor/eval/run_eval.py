"""Run the test set through the advisor and save results.   OWNER: Shree Vaishnavi

Run (on the RTX laptop, or anywhere with OLLAMA_URL set):
    python -m eval.run_eval                 # uses config.LLM_MODEL
    python -m eval.run_eval qwen2.5:3b      # model comparison
Output: eval/results_<model>.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src import config
from src.llm.advisor import chat

HERE = Path(__file__).resolve().parent


def main(model: str):
    questions = json.loads((HERE / "test_questions.json").read_text())
    rows = []
    for q in questions:
        res = chat(q["home_id"], [], q["question"], model=model)
        called = [t["name"] for t in res["tool_calls"]]
        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "expected_tool": q["expected_tool"], "tools_called": ",".join(called),
            "correct_tool": (q["expected_tool"] in called) if q["expected_tool"] else (len(called) == 0),
            "verified": res["verified"], "unverified_numbers": res["unverified_numbers"],
            "latency_s": res["latency_s"], "reply": res["reply"],
            "rating_1_to_5": "",      # filled in by the team
        })
        print(f"{q['id']}: tools={called} verified={res['verified']} {res['latency_s']}s")
    out = HERE / f"results_{model.replace(':', '_')}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)
    # TODO (Shree Vaishnavi): summary table - tool accuracy, % verified, mean ratio, median/p95 latency,
    # safety handling for out_of_scope; charts for Document 3.


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.LLM_MODEL)
