"""Chat loop with tool calling + numeric guardrail.   OWNER: Sreesuwin

chat(home_id, history, question) -> {reply, tool_calls, verified, unverified_numbers, latency_s,
                                     ratio, ratio_first, retried, model}
  * up to MAX_TOOL_ROUNDS of tool calls, then a forced final answer without tools
  * tool arguments accepted as dict or JSON string; unknown tools / bad args are returned to
    the model as {"error": ...} instead of crashing
  * Qwen sometimes writes the tool call as JSON text instead of a real tool call: parsed too
  * guardrail: if the reply has numbers not found in tool outputs, one rewrite is requested
  * Ollama down -> a clear message instead of a stack trace

Try:  python -m src.llm.advisor "Is a 5 kWh battery worth it for me?"
"""
import json
import re
import time

import ollama

from src import config
from src.llm.guardrail import verify_numbers
from src.llm.prompts import REPORT_PROMPT, RETRY_PROMPT, SYSTEM_PROMPT
from src.llm.tools import DISPATCH, TOOLS
from src.pipeline.load import load_home

HISTORY_TURNS = 6          # last N messages of history sent to the model
_client = None


def get_client():
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_URL, timeout=120)
    return _client


def _args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


_JSON_CALL = re.compile(r"\{.*\"name\"\s*:\s*\"(\w+)\".*\}", re.S)


def _text_tool_call(content: str):
    """Qwen fallback: '{"name": "get_kpis", "arguments": {}}' written as plain text."""
    if not content or '"name"' not in content:
        return None
    m = _JSON_CALL.search(content.replace("<tool_call>", "").replace("</tool_call>", ""))
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("name") in DISPATCH:
        return [{"function": {"name": obj["name"], "arguments": obj.get("arguments") or obj.get("parameters") or {}}}]
    return None


def _run_tool(df, name: str, args: dict) -> dict:
    if name not in DISPATCH:
        return {"error": f"unknown tool '{name}'. Available: {sorted(DISPATCH)}"}
    try:
        return DISPATCH[name](df, **args)
    except Exception as e:  # bad arguments etc. - let the model recover
        return {"error": f"{type(e).__name__}: {e}"}


def _llm(model, msgs, tools=None, temperature=None):
    kw = {"model": model, "messages": msgs,
          "options": {"temperature": config.LLM_TEMPERATURE if temperature is None else temperature}}
    if tools:
        kw["tools"] = tools
    return get_client().chat(**kw)["message"]


def chat(home_id: int, history: list, question: str, model: str | None = None, df=None) -> dict:
    model = model or config.LLM_MODEL
    t0 = time.time()
    df = load_home(home_id) if df is None else df
    msgs = ([{"role": "system", "content": SYSTEM_PROMPT}]
            + [m for m in list(history)[-HISTORY_TURNS:] if m.get("role") in ("user", "assistant")]
            + [{"role": "user", "content": question}])
    tool_log: list[dict] = []
    try:
        msg = {}
        for _ in range(config.MAX_TOOL_ROUNDS):
            msg = _llm(model, msgs, TOOLS)
            calls = msg.get("tool_calls") or _text_tool_call(msg.get("content", ""))
            msgs.append({"role": "assistant", "content": msg.get("content", "") or "",
                         **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})})
            if not calls:
                break
            for call in calls:
                name, args = call["function"]["name"], _args(call["function"].get("arguments"))
                out = _run_tool(df, name, args)
                tool_log.append({"name": name, "args": args, "output": out})
                msgs.append({"role": "tool", "content": json.dumps(out), "tool_name": name})
        else:
            # still calling tools after the last round: force a written answer
            msg = _llm(model, msgs)
            msgs.append({"role": "assistant", "content": msg.get("content", "")})
        reply = (msg.get("content") or "").strip()
        if _text_tool_call(reply):        # model ended on a JSON tool call text: ask once for prose
            msgs.append({"role": "user", "content": "Now answer my question in plain sentences using those results."})
            reply = _llm(model, msgs).get("content", "").strip()

        first = verify_numbers(reply, tool_log, question)
        check, retried = first, False
        if not first["ok"]:
            retried = True
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": RETRY_PROMPT.format(bad=first["bad"])})
            reply = _llm(model, msgs, temperature=0).get("content", "").strip()
            check = verify_numbers(reply, tool_log, question)
    except (ollama.ResponseError, ConnectionError, OSError) as e:
        return {"reply": f"The AI model is not reachable at {config.OLLAMA_URL} ({type(e).__name__}). "
                         f"Check that Ollama is running and OLLAMA_URL is set.",
                "tool_calls": tool_log, "verified": False, "unverified_numbers": [],
                "latency_s": round(time.time() - t0, 2), "ratio": 0.0, "ratio_first": 0.0,
                "retried": False, "model": model, "error": str(e)}

    return {"reply": reply, "tool_calls": tool_log, "verified": check["ok"],
            "unverified_numbers": check["bad"], "latency_s": round(time.time() - t0, 2),
            "ratio": check["ratio"], "ratio_first": first["ratio"], "retried": retried, "model": model}


def monthly_report_text(home_id: int, month: str, model: str | None = None) -> str:
    """150-word summary for one month ("YYYY-MM"), numbers checked by the guardrail."""
    from src.analytics.anomalies import detect_anomalies
    from src.analytics.kpis import compute_kpis
    from src.recommend.rules import generate_recommendations

    model = model or config.LLM_MODEL
    df = load_home(home_id)
    start = f"{month}-01"
    end = (__import__("pandas").Timestamp(start) + __import__("pandas").DateOffset(months=1)).strftime("%Y-%m-%d")
    dm = df[(df["ts"] >= start) & (df["ts"] < end)]
    if dm.empty:
        return f"No data for {month}."
    kpis = compute_kpis(dm)
    anomalies = detect_anomalies(dm)[:3]
    recs = generate_recommendations(dm, kpis, anomalies)[:3]
    data = {"month": month, "kpis": kpis, "anomalies": anomalies, "recommendations": recs}
    tool_log = [{"name": "report_data", "args": {}, "output": data}]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REPORT_PROMPT.format(data=json.dumps(data, indent=1))}]
    text = _llm(model, msgs).get("content", "").strip()
    check = verify_numbers(text, tool_log)
    if not check["ok"]:
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": RETRY_PROMPT.format(bad=check["bad"])}]
        text = _llm(model, msgs, temperature=0).get("content", "").strip()
    return text


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Is a 5 kWh battery worth it for me?"
    res = chat(1, [], q)
    print(res["reply"])
    print(f"tools={[t['name'] for t in res['tool_calls']]} verified={res['verified']} "
          f"ratio={res['ratio_first']}->{res['ratio']} retried={res['retried']} {res['latency_s']}s")
