"""Numeric guardrail: every number in a reply must trace back to a tool output.   OWNER: Sreesuwin

STUB: always passes. Real version (README 4.4):
 1. numbers in reply: re.findall(r"\\d[\\d,]*\\.?\\d*", reply), strip commas
 2. allowed = all numbers in tool outputs (walk recursively) + numbers in the question
              + 0-24 (hours, counts) + calendar years
 3. pass if within ±1% or ±0.05 of an allowed number, or equal after rounding to 0/1 decimals
"""


def verify_numbers(reply: str, tool_log: list[dict], question: str = "") -> dict:
    # STUB
    return {"ok": True, "bad": [], "ratio": 1.0}
