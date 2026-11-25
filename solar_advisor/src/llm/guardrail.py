"""Numeric guardrail: every number in a reply must trace back to a tool output.   OWNER: Sreesuwin

How it works
 1. Pull every number out of the reply ("₹6,052", "52.6%", "24.8 years" -> 6052, 52.6, 24.8).
 2. Build the allowed set: every number in the tool outputs and tool arguments (walked
    recursively, including numbers inside strings such as dates and reasons), every number
    in the user's question, small integers 0-24 (hours, counts, list numbering) and years.
 3. A reply number passes if it is within max(0.05, 1%) of an allowed number, or equals an
    allowed number rounded to 0 or 1 decimals (the model may say "25 years" for 24.8).
 4. ok = no unverified numbers; ratio = verified / total (our groundedness metric).

Anything the model computed itself (e.g. 6052 / 12 = 504 per month) is NOT in the
allowed set and is reported in "bad", so advisor.chat() asks the model to rewrite.
"""
import re

NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
SMALL_INTS = set(range(0, 25))
YEARS = set(range(2000, 2036))


def extract_numbers(text: str) -> list[float]:
    """All numbers in a piece of text. Thousands separators are removed."""
    out = []
    for m in NUM_RE.findall(text or ""):
        s = m.replace(",", "").rstrip(".")
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _walk(obj, acc: list):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        acc.extend(extract_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, acc)


def allowed_numbers(tool_log: list[dict], question: str = "") -> list[float]:
    acc: list[float] = []
    for entry in tool_log or []:
        _walk(entry.get("output"), acc)
        _walk(entry.get("args"), acc)
    acc.extend(extract_numbers(question))
    return acc


def _matches(x: float, allowed: list[float]) -> bool:
    if x in SMALL_INTS or (x.is_integer() and int(x) in YEARS):
        return True
    for a in allowed:
        if abs(x - a) <= max(0.05, 0.01 * abs(a)):
            return True
        if x == round(a) or x == round(a, 1):
            return True
    return False


def verify_numbers(reply: str, tool_log: list[dict], question: str = "") -> dict:
    nums = extract_numbers(reply)
    if not nums:
        return {"ok": True, "bad": [], "ratio": 1.0}
    allowed = allowed_numbers(tool_log, question)
    bad = [x for x in nums if not _matches(x, allowed)]
    bad_clean = sorted({int(x) if x.is_integer() else x for x in bad})
    return {"ok": not bad, "bad": bad_clean, "ratio": round((len(nums) - len(bad)) / len(nums), 2)}
