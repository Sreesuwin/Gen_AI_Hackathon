"""Chat loop with tool calling.   OWNER: Sreesuwin

This is a working first version: Ollama + tools + one guardrail retry. With the stub
modules it runs end to end today. Improve prompts, error handling, history trimming.
"""
import json
import time

import ollama

from src import config
from src.llm.guardrail import verify_numbers
from src.llm.prompts import RETRY_PROMPT, SYSTEM_PROMPT
from src.llm.tools import DISPATCH, TOOLS
from src.pipeline.load import load_home

_client = None


def get_client():
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_URL)
    return _client


def chat(home_id: int, history: list, question: str, model: str | None = None) -> dict:
    model = model or config.LLM_MODEL
    df = load_home(home_id)
    t0 = time.time()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history) + [{"role": "user", "content": question}]
    tool_log = []
    msg = {}
    for _ in range(config.MAX_TOOL_ROUNDS):
        r = get_client().chat(model=model, messages=msgs, tools=TOOLS,
                              options={"temperature": config.LLM_TEMPERATURE})
        msg = r["message"]
        msgs.append(msg)
        if not msg.get("tool_calls"):
            break
        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], dict(call["function"]["arguments"] or {})
            try:
                out = DISPATCH[name](df, **args)
            except Exception as e:                     # bad tool name / args: tell the model, don't crash
                out = {"error": f"{type(e).__name__}: {e}"}
            tool_log.append({"name": name, "args": args, "output": out})
            msgs.append({"role": "tool", "content": json.dumps(out)})
    reply = msg.get("content", "")
    check = verify_numbers(reply, tool_log, question)
    if not check["ok"]:
        msgs.append({"role": "user", "content": RETRY_PROMPT.format(bad=check["bad"])})
        reply = get_client().chat(model=model, messages=msgs, options={"temperature": 0})["message"]["content"]
        check = verify_numbers(reply, tool_log, question)
    return {"reply": reply, "tool_calls": tool_log, "verified": check["ok"],
            "unverified_numbers": check["bad"], "latency_s": round(time.time() - t0, 2)}


def monthly_report_text(home_id: int, month: str) -> str:
    """TODO (Sreesuwin): KPIs + recommendations + anomalies for `month` -> REPORT_PROMPT -> guardrail."""
    raise NotImplementedError("monthly_report_text: Sreesuwin")


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Is a 5 kWh battery worth it for me?"
    res = chat(1, [], q)
    print(res["reply"])
    print("tools:", [t["name"] for t in res["tool_calls"]], "| verified:", res["verified"], "|", res["latency_s"], "s")
