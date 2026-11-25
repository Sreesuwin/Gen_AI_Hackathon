"""Guardrail unit tests (no LLM needed).   OWNER: Sreesuwin"""
from src.llm.guardrail import extract_numbers, verify_numbers

BATTERY = [{"name": "simulate_battery", "args": {"capacity_kwh": 5},
            "output": {"capacity_kwh": 5, "power_kw": 2.5, "import_avoided_kwh": 1210.4, "savings_inr_year": 6052.0,
                       "self_consumption_pct_before": 52.6, "self_consumption_pct_after": 81.3,
                       "cost_inr": 150000, "payback_years": 24.8}}]


def test_extract_formats():
    assert extract_numbers("₹6,052 a year, 52.6% now, payback 24.8 years.") == [6052.0, 52.6, 24.8]
    assert extract_numbers("costs ₹1,50,000") == [150000.0]          # Indian grouping
    assert extract_numbers("qwen2.5 model") == []                     # digits glued to a word are ignored


def test_exact_numbers_pass():
    r = verify_numbers("A 5 kWh battery saves ₹6,052 a year and avoids 1,210.4 kWh. "
                       "Self-consumption rises from 52.6% to 81.3%. It costs ₹150,000, payback 24.8 years.", BATTERY)
    assert r == {"ok": True, "bad": [], "ratio": 1.0}


def test_rounding_passes():
    r = verify_numbers("About ₹6,050 a year, payback about 25 years, self-consumption about 81%.", BATTERY)
    assert r["ok"], r


def test_computed_number_is_caught():
    # 6052 / 12 = 504.33 is a calculation the model did itself
    r = verify_numbers("That is about ₹504 per month.", BATTERY)
    assert not r["ok"] and r["bad"] == [504]


def test_invented_number_is_caught():
    r = verify_numbers("Payback is 12 years and you save ₹9,000.", BATTERY)
    assert not r["ok"] and 9000 in r["bad"]


def test_question_numbers_allowed():
    r = verify_numbers("Driving 40 km a day adds load.", [], question="What if I drive 40 km a day?")
    assert r["ok"]


def test_numbers_inside_strings_allowed():
    log = [{"name": "get_anomalies", "args": {}, "output": {"anomalies": [
        {"date": "2013-03-14", "reason": "Night load 3x higher than usual", "value": 0.92}]}}]
    assert verify_numbers("On 2013-03-14 your night load hit 0.92 kW, 3x usual.", log)["ok"]


def test_no_numbers():
    assert verify_numbers("I can't help with wiring; please call a licensed electrician.", [])["ok"]


def test_ordinals_are_labels():
    assert extract_numbers("between the 10th and 90th percentile, 4.66 kWh on the 1st day") == [4.66]
