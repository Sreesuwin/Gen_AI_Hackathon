# SolarSense – Solar Energy Optimization and Advisory Assistant

Team SolarSense · PS 15 · Generative AI Semester-End Hackathon · 25 Nov 2026

This README is the single source of truth for the project. It is written so that it can be
pasted into (or read by) an AI coding assistant as its knowledge base. If you are an AI
assistant helping a team member, read **section 0** first and follow it strictly.

---

## 0. Instructions for AI coding assistants

You are helping ONE team member build ONE module of a 4-person project. Others are building
the other modules at the same time, against the contract in section 4.

1. **Only edit files in your owner's folder** (see the ownership table in section 2). Never
   edit `src/contracts.py`, `src/config.py` or another person's module. If a change there
   seems necessary, stop and tell the user to agree it with the team.
2. **Keep every function signature and every returned key exactly as in section 4.**
   `tests/test_contracts.py` must still pass after your change: run `pytest -q`.
3. Returned dicts: numbers rounded to **at most 2 decimals**, **plain Python types only**
   (`float(x)`, `int(x)`, never numpy/pandas scalars), units in key names
   (`_kwh`, `_kw`, `_inr`, `_pct`, `_years`).
4. Read tariffs, costs, paths and the model name from `src/config.py`. Never hard-code them.
5. Dependencies: only what is in `requirements.txt` (plus `torch` + `chronos-forecasting`
   for forecasting). Python **3.11**. Ask the user before adding anything else.
6. Only **Sreesuwin's laptop (RTX 4070, 8 GB)** has a GPU. Everything else must run on
   CPU. The LLM is reached over the network at `config.OLLAMA_URL`.
7. Prefer simple, readable pandas code over clever code. This is a one-day hackathon:
   working and correct beats complete. Do the "Review 2" items before "Review 3" items.
8. The LLM must never calculate numbers. All numbers come from Python functions (tools);
   the LLM only explains them. Do not add LLM calls inside analytics/recommendation code.
9. Ausgrid data quirks are listed in section 3.3. Read them before touching the data.

---

## 1. What we are building

Rooftop solar owners use only about half of the solar energy they generate. Solar peaks
around noon, household demand peaks in the evening, so surplus is exported cheaply and
evening power is bought back at full price. Our analysis of 300 real homes (FY 2012–13):

| Finding | Value |
|---|---|
| Solar used inside the home (self-consumption) | 52.6% |
| Demand met by solar (self-sufficiency) | 18.0% |
| Solar peak / demand peak | 12:00–13:00 (0.86 kW) / 18:00–19:00 (1.06 kW) |
| Night load, median home / top 10% | 0.26 kW / 0.78 kW (3×) |

**Solution:** numbers come from code, words come from the LLM.

```
Data (Ausgrid CSV / user CSV)
  → M1 Data pipeline      (clean, reshape, import/export)          Shashikiran
  → M2 Energy analytics   (KPIs, anomalies, forecast)              Shashikiran
  → M3 Recommendations    (rules, battery/PV/EV/load-shift sims)   Divya
  → M4 GenAI advisor      (Qwen2.5-7B via Ollama, tool calling,    Sreesuwin
                           numeric guardrail)
  → Streamlit UI          (dashboard, chat, report)                Divya
Evaluation + documents                                             Shree Vaishnavi
```

Features: energy health report · load-shift planner · inefficiency finder ·
battery and panel sizing · what-if chat · monthly PDF report.

**GenAI technique:** LLM with tool calling (light agentic). Not RAG (answers are numbers
computed per home, no document corpus), not OCR/multimodal (inputs are structured meter
data), not fine-tuning (no labelled advice data). A numeric guardrail checks every number
in a reply against tool outputs and forces one rewrite if a number cannot be traced.

---

## 2. Team, ownership and hardware

| Member | Role | Owns (folders) | Machine |
|---|---|---|---|
| Shashikiran Sekar | Data and Analytics | `src/pipeline/`, `src/analytics/` | CPU laptop |
| Sreesuwin S R | GenAI Lead | `src/llm/`, Ollama server | **RTX 4070 laptop (8 GB) – only GPU** |
| Divya R | App and Integration | `src/recommend/`, `src/app/`, Dockerfile, docker-compose.yml | CPU laptop |
| Shree Vaishnavi M | Evaluation and Docs | `eval/`, documents | CPU laptop |

Shared (change only with team agreement): `src/config.py`, `src/contracts.py`, `tests/`.

### 2.1 The LLM server

`qwen2.5:7b` (4-bit, 4.7 GB) runs in Ollama on Sreesuwin's laptop. Measured: ~54 tokens/s,
first call after idle ~10–50 s (model load), tool calling works. Fallback: `qwen2.5:3b`.

To let other laptops use it (Sreesuwin, once):
```bash
sudo systemctl edit ollama
#   add:  [Service]
#         Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl restart ollama
hostname -I            # share this IP with the team (was 172.22.66.56)
```
Everyone else:
```bash
export OLLAMA_URL=http://<sreesuwin-ip>:11434
curl $OLLAMA_URL/api/version          # should print {"version": ...}
```
Only Shashikiran, Divya and Shree Vaishnavi's chat/eval calls need this; the analytics and
recommendation code never calls the LLM.

---

## 3. Setup

### 3.1 Environment (every laptop)
```bash
git clone https://github.com/Sreesuwin/Gen_AI_Hackathon.git && cd Gen_AI_Hackathon/solar_advisor
pip install uv                                   # or: pipx install uv
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
pytest -q                                        # 10 contract tests must pass
```
Shashikiran only (forecasting, CPU build keeps the download small):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install chronos-forecasting
```

### 3.2 Data
- Raw data is **not in git** (190 MB). Copy the three files
  `Solar home 2010-2011.csv`, `Solar home 2011-2012.csv`, `Solar home 2012-2013.csv`
  from `Ausgrid_solar_home_data.zip` into `data/raw/`.
- `data/sample/sample_home.parquet` **is in git**: home 1 (3.78 kW PV, postcode 2076),
  FY 2012–13, 17,520 half-hour rows in the contract schema. Recreate with
  `python scripts/make_sample.py [home_id] [year]`.
- Until `data/processed/halfhourly.parquet` exists, `load_home()` returns the sample for any
  home id. After Shashikiran runs `python -m src.pipeline.build`, everything uses real data
  automatically. Others can copy his two parquet files into `data/processed/`.

### 3.3 Ausgrid data facts and quirks (read before touching data)
- Source: Ausgrid Solar Home Electricity Data, NSW Australia, CC BY 3.0. 300 homes,
  1 Jul 2010 – 30 Jun 2013, gross-metered rooftop PV.
- One row per **home × category × day**; columns: `Customer, Generator Capacity, Postcode,
  Consumption Category, date, 0:30, 1:00, …, 23:30, 0:00` (+ `Row Quality` in 2011-12 and 2012-13).
- **Line 1 is a title** → `pd.read_csv(path, skiprows=1)`.
- **Use `keep_default_na=False`**: the quality flag is the literal text `NA`, which pandas
  would otherwise turn into NaN, the same as an empty (good) row. Drop rows where
  `Row Quality == "NA"` (531 rows in total).
- **Date formats differ:** 2010-11 `"%d-%b-%y"` (1-Jul-10); 2011-12 and 2012-13 `"%d/%m/%Y"` (1/07/2011).
- The 48 values are kWh per half-hour. Slot *i* (0-based, columns 5–52) covers the interval
  starting at `date + i × 30 min`; the last column `0:00` is 23:30–24:00 of the same date.
- Categories: `GG` = gross solar generation, `GC` = general consumption, `CL` = controlled load
  (off-peak hot water, only 139 of 300 homes). Consumption = GC + CL.
- **Gross metering**: import/export are not measured. Per half-hour:
  `self = min(gen, cons)`, `import = max(cons − gen, 0)`, `export = max(gen − cons, 0)`.
- **Impossible readings:** `gen_kwh > pv_kw × 0.5 × 1.05` (more than the panel can make in
  30 min) → `valid = False` (9,184 half-hours). `load_home()` returns valid rows only.
- kWh per half-hour × 2 = average kW over that half-hour.
- Data is Australian and 2010–13; we treat it as a benchmark and show money in ₹ with
  configurable tariffs. Patterns (noon solar peak, evening demand peak, standby load) match India.

---

## 4. The contract (all modules follow this)

Defined in `src/contracts.py`, checked by `tests/test_contracts.py`.

### 4.1 Half-hourly DataFrame (`load_home` output, `data/processed/halfhourly.parquet`)

| column | type | meaning |
|---|---|---|
| home_id | int | Ausgrid customer 1–300 (0 = user upload) |
| ts | datetime | start of the 30-min interval |
| gen_kwh | float | solar generation |
| gc_kwh | float | general consumption |
| cl_kwh | float | controlled load (0 if none) |
| cons_kwh | float | gc_kwh + cl_kwh |
| self_kwh | float | min(gen_kwh, cons_kwh) |
| import_kwh | float | max(cons_kwh − gen_kwh, 0) |
| export_kwh | float | max(gen_kwh − cons_kwh, 0) |
| valid | bool | False for flagged/impossible readings |

Homes table (`list_homes`, `homes.parquet`): `home_id, pv_kw, postcode, has_cl, n_days`.

### 4.2 Function signatures

```python
# src/pipeline/load.py, src/pipeline/build.py           (Shashikiran)
list_homes() -> pd.DataFrame
load_home(home_id: int, start=None, end=None) -> pd.DataFrame
load_user_csv(file, pv_kw: float) -> pd.DataFrame            # cols timestamp, generation_kwh, consumption_kwh

# src/analytics/kpis.py, anomalies.py, forecast.py       (Shashikiran)
compute_kpis(df) -> dict
daily_profile(df) -> pd.DataFrame                            # hour, gen_kw, cons_kw
monthly_summary(df) -> pd.DataFrame                          # month, gen_kwh, cons_kwh, import_kwh, export_kwh, self_consumption_pct
detect_anomalies(df) -> list[dict]
forecast(df, horizon_days: int = 14, target: str = "cons") -> dict      # target "cons" | "gen"

# src/recommend/simulator.py, rules.py                   (Divya)
simulate_battery(df, capacity_kwh: float, power_kw: float | None = None) -> dict
simulate_extra_pv(df, extra_kw: float) -> dict
simulate_load_shift(df, kwh_per_day: float, from_hours=(18, 22), to_hours=(10, 14)) -> dict
simulate_ev(df, km_per_day: float, charge_hours=(10, 15)) -> dict
generate_recommendations(df, kpis: dict, anomalies: list[dict]) -> list[dict]   # sorted by inr_saved_month desc

# src/llm/advisor.py, guardrail.py, tools.py              (Sreesuwin)
chat(home_id: int, history: list, question: str, model: str | None = None) -> dict
monthly_report_text(home_id: int, month: str) -> str          # month "YYYY-MM"
verify_numbers(reply: str, tool_log: list[dict], question: str = "") -> dict
TOOLS: list[dict]        # Ollama tool schemas
DISPATCH: dict[str, Callable[[df, **args], dict]]
```

### 4.3 Returned dict shapes (keys are mandatory; values here are examples)

```python
# compute_kpis
{"period": "2012-07-01 to 2013-06-30", "days": 365,
 "gen_kwh": 2180.4, "cons_kwh": 6391.2, "self_kwh": 1146.3, "import_kwh": 5244.9, "export_kwh": 1034.1,
 "self_consumption_pct": 52.6,      # self_kwh / gen_kwh * 100
 "self_sufficiency_pct": 17.9,      # self_kwh / cons_kwh * 100
 "baseload_kw": 0.26,               # median over days of mean 01:00-05:00 consumption, in kW
 "peak_kw": 3.8,                    # 99th percentile of cons_kwh*2
 "peak_hour": 18,                   # hour of day with highest average consumption
 "avg_daily_gen_kwh": 5.97, "avg_daily_cons_kwh": 17.51,
 "grid_cost_inr": 41959.2,          # import_kwh * IMPORT_TARIFF
 "export_credit_inr": 3102.3,       # export_kwh * EXPORT_TARIFF
 "net_bill_inr": 38856.9}

# detect_anomalies -> list of
{"date": "2013-03-14", "type": "high_night_load",    # | "consumption_spike" | "low_generation"
 "value": 0.92, "expected": 0.31, "unit": "kW", "score": 0.81,
 "reason": "Night load 3x higher than this home's usual level"}

# forecast
{"target": "cons", "horizon_days": 14, "dates": ["2013-07-01", ...],
 "p10": [...], "p50": [...], "p90": [...],          # daily kWh, one per date
 "total_p50_kwh": 241.5, "method": "seasonal_baseline"}   # | "chronos-2" | "chronos-bolt"

# simulate_battery
{"capacity_kwh": 5, "power_kw": 2.5, "import_avoided_kwh": 1210.4, "savings_inr_year": 6052.0,
 "self_consumption_pct_before": 52.6, "self_consumption_pct_after": 81.3,
 "cost_inr": 150000, "payback_years": 24.8}

# simulate_extra_pv
{"extra_kw": 2, "extra_gen_kwh_year": 1350.0, "import_avoided_kwh": 410.2, "extra_export_kwh": 939.8,
 "savings_inr_year": 6101.0, "cost_inr": 100000, "payback_years": 16.4, "self_sufficiency_pct_after": 24.3}

# simulate_load_shift
{"kwh_per_day": 1.5, "shifted_kwh_year": 438.0, "savings_inr_year": 2190.0, "self_consumption_pct_after": 72.7}

# simulate_ev
{"km_per_day": 40, "ev_kwh_year": 2190.0, "solar_share_pct": 38.5,
 "extra_import_kwh_year": 1347.2, "extra_cost_inr_year": 10777.6}

# generate_recommendations -> list of
{"id": "R2", "title": "Cut always-on night load", "category": "inefficiency",   # | "utilization" | "planning"
 "action": "Check devices left on overnight (old fridge, pumps, chargers).",
 "kwh_saved_month": 86.4, "inr_saved_month": 691.2, "effort": "low",          # | "medium" | "high"
 "evidence": {"baseload_kw": 0.62, "median_home_baseload_kw": 0.26}}

# chat
{"reply": "...", "tool_calls": [{"name": "simulate_battery", "args": {...}, "output": {...}}],
 "verified": True, "unverified_numbers": [], "latency_s": 3.2}

# verify_numbers
{"ok": True, "bad": [], "ratio": 1.0}       # ratio = verified numbers / numbers in reply
```

### 4.4 Shared config (`src/config.py`, override any value with an env var of the same name)

`IMPORT_TARIFF = 8.0` ₹/kWh · `EXPORT_TARIFF = 3.0` ₹/kWh · `BATTERY_COST_PER_KWH = 30000` ₹ ·
`PV_COST_PER_KW = 50000` ₹ · `BATTERY_EFFICIENCY = 0.90` · `EV_KWH_PER_KM = 0.15` ·
`DATASET_MEDIAN_BASELOAD_KW = 0.26` · `OLLAMA_URL` · `LLM_MODEL = "qwen2.5:7b"` ·
`LLM_FALLBACK_MODEL = "qwen2.5:3b"` · `MAX_TOOL_ROUNDS = 4` · paths.

### 4.5 Current state of each file

Every module exists with the right signature. **STUB** = returns fixed example values from 4.3.

| File | State |
|---|---|
| `scripts/make_sample.py` | done (`reshape()` is reusable for the full pipeline) |
| `src/pipeline/load.py` | `list_homes`, `load_home` done (fall back to sample); `load_user_csv` TODO |
| `src/pipeline/build.py` | TODO |
| `src/analytics/kpis.py` | `compute_kpis` STUB; `daily_profile`, `monthly_summary` done |
| `src/analytics/anomalies.py` | STUB |
| `src/analytics/forecast.py` | STUB |
| `src/recommend/simulator.py` | all STUB |
| `src/recommend/rules.py` | STUB |
| `src/llm/tools.py` | done (7 tools + DISPATCH) |
| `src/llm/advisor.py` | `chat` working (tested: called `simulate_battery`, quoted its numbers); `monthly_report_text` TODO |
| `src/llm/guardrail.py` | STUB (always ok) |
| `src/llm/prompts.py` | first version |
| `src/app/streamlit_app.py` | Overview + Ask tabs working; other tabs TODO |
| `eval/run_eval.py` | working; summary TODO. `eval/test_questions.json` has 3 of 25 |
| `tests/test_contracts.py` | 10 tests, all pass on stubs |

---

## 5. Shashikiran Sekar – Data Pipeline and Analytics

**Review 2 (must have):** pipeline, real KPIs, anomalies, baseline forecast.
**Review 3:** Chronos forecast + MAPE vs baseline, `load_user_csv`.

### 5.1 Pipeline – `src/pipeline/build.py` (`python -m src.pipeline.build`)
1. For each of the 3 files: `pd.read_csv(path, skiprows=1, keep_default_na=False)`; drop
   `Row Quality == "NA"`; `reshape(raw, config.DATE_FORMATS[year])` from `scripts/make_sample.py`
   (already handles melt, pivot GG/GC/CL, derived columns and `valid`).
2. Concatenate years; drop duplicate `(home_id, ts)`.
3. Save `config.HALFHOURLY_FILE` (~15.8 M rows; use `df.to_parquet(..., index=False)`;
   cast floats to float32 if memory is tight) and `config.HOMES_FILE`
   (`home_id, pv_kw, postcode, has_cl, n_days`).
4. Memory tip: reshape file by file, or home by home in chunks of 50, then concat.
5. `load_home()` already reads with a `home_id` filter, so nothing else changes.

### 5.2 KPIs – `src/analytics/kpis.py`
Replace the `compute_kpis` stub. Definitions are in the comments of 4.3. `days` =
number of distinct dates; `period` = `"<first date> to <last date>"`. Use `config` tariffs.
Round everything with `round(float(x), 2)`.

### 5.3 Anomalies – `src/analytics/anomalies.py`
1. Daily features: `daily_cons_kwh`, `night_load_kw` (mean 01:00–05:00 × 2), `peak_kw`
   (max cons_kwh × 2), `evening_share` (18:00–22:00 share of daily cons), `yield_ratio`
   (daily gen ÷ 90th percentile of daily gen in the same calendar month for this home).
2. `IsolationForest(contamination=0.03, random_state=42)` on the 4 consumption features
   (standardise first). Flagged days only.
3. For each flagged day, z-score each feature vs the home's median/std; the largest names
   the type: `night_load_kw` → `high_night_load`; `daily_cons_kwh` or `peak_kw` → `consumption_spike`.
   `value` = that feature, `expected` = home median, `score` = normalised IF score 0–1,
   `reason` = plain sentence with the ratio ("Night load 3x higher than usual").
4. Panel rule: 14-day rolling median of `yield_ratio` < 0.75 → one `low_generation`
   anomaly per run (date = run start), reason "possible dust, shading or fault".
5. Return newest first, at most ~20 items.

### 5.4 Forecast – `src/analytics/forecast.py`
- Daily totals of `cons_kwh` (target "cons") or `gen_kwh` ("gen").
- **Baseline (Review 2):** each future day = mean of the same weekday over the last 4 weeks;
  p10/p90 = min/max (or 10th/90th pct) of those values. `method = "seasonal_baseline"`.
- **Chronos (Review 3):** context = last 365 daily values; quantiles 0.1/0.5/0.9; load the
  pipeline once (module-level cache) on CPU. Try `amazon/chronos-2` (see the
  `chronos-forecasting` README for the Chronos-2 pipeline class and `predict` call); if it
  causes trouble use `amazon/chronos-bolt-small` via `BaseChronosPipeline` and set
  `method = "chronos-bolt"`.
- Evaluation helper for Shree Vaishnavi: hold out the last 30 days for 20 homes, report MAPE
  of Chronos vs baseline.

### 5.5 `load_user_csv` (Review 3)
Columns `timestamp, generation_kwh, consumption_kwh`; any 15–60 min resolution →
resample to 30 min (sum); `home_id = 0`; `cl_kwh = 0`; derive the rest; `valid` from `pv_kw`.

---

## 6. Divya R – Recommendation Engine and App

**Review 2:** battery + load-shift simulators, rules R1/R2/R5, Overview/Recommendations/Ask tabs.
**Review 3:** all simulators and rules, all tabs, PDF report, Docker.

### 6.1 Simulators – `src/recommend/simulator.py`
Step through half-hours in time order (plain loop, ~17,520 rows/year is fast). Scale yearly
figures by `365 / days_in_data`. Use `config` costs and tariffs.

Battery (default `power_kw = capacity / 2`, `step = power_kw * 0.5` kWh per half-hour):
```python
soc = 0.0
for gen, cons in zip(df.gen_kwh.values, df.cons_kwh.values):
    surplus = gen - cons
    if surplus > 0:
        charge = min(surplus, step, (cap - soc) / eff)
        soc += charge * eff
        exp_new += surplus - charge
    else:
        dis = min(-surplus, step, soc)
        soc -= dis
        imp_new += -surplus - dis
savings = (imp_old - imp_new) * IMPORT_TARIFF - (exp_old - exp_new) * EXPORT_TARIFF
payback = cap * BATTERY_COST_PER_KWH / savings          # guard savings <= 0 -> payback 99.0
self_consumption_after = (gen_total - exp_new) / gen_total * 100
```
- **Extra PV:** `gen × (pv_kw + extra_kw) / pv_kw` (pv_kw from `list_homes()`), recompute
  self/import/export per half-hour.
- **Load shift:** per day, `shift = min(kwh_per_day, import in from_hours, export in to_hours)`;
  savings = shift × (IMPORT_TARIFF − EXPORT_TARIFF).
- **EV:** add `km_per_day × EV_KWH_PER_KM` spread evenly over `charge_hours` each day;
  `solar_share_pct` = part of EV energy met by surplus; extra import × tariff = extra cost.

### 6.2 Rules – `src/recommend/rules.py`
Each rule: check condition → return a dict with all `RECOMMENDATION_KEYS` and an `evidence`
dict of the numbers used. `generate_recommendations` runs all, drops `None`, sorts by
`inr_saved_month` descending.

| ID | Condition | Action | Impact |
|---|---|---|---|
| R1 Load shift | self_consumption_pct < 60 and evening import > 2 kWh/day | Run washing, dishwasher, pumps, water heating 10 am–2 pm | `simulate_load_shift(df, 1.5)` /12 per month |
| R2 Night load | baseload_kw > 0.4 or > 1.5 × 0.26 | Find always-on devices | (baseload − 0.26) × 24 × 30 kWh/month × tariff |
| R3 Panel check | any `low_generation` anomaly in last 60 days | Clean or inspect panels | expected − actual generation |
| R4 Hot water | has CL and export > 3 kWh/day | Move water-heater timer to midday | min(night CL kWh, export kWh) per day |
| R5 Battery | best of 3/5/10 kWh has payback < 12 years (else still report best, effort "high") | Consider that size | `simulate_battery` |
| R6 Peak spikes | peak_kw > 2.5 × average evening kW | Stagger heavy appliances | peak reduction estimate |

### 6.3 App – `src/app/streamlit_app.py` (`streamlit run src/app/streamlit_app.py`)
- Sidebar: home picker (preselect 5 good demo homes), CSV upload (Review 3), tariff inputs,
  model switch 7B/3B (exists).
- Tabs: **Overview** (exists: 4 KPI cards, average-day chart; add monthly bars and shaded
  self-used area) · **Inefficiencies** (anomaly table, daily chart with flagged days) ·
  **Forecast** (p50 line + p10–p90 band, method label) · **Recommendations** (ranked cards,
  evidence in expander) · **Planner** (battery slider 0–15, extra PV slider, EV km input →
  live simulator results) · **Ask SolarSense** (exists) · **Report** (ReportLab PDF: KPI table,
  chart image, `monthly_report_text`, recommendations).
- Cache: `@st.cache_data` for data/KPIs, `@st.cache_resource` for clients/models.

### 6.4 Docker (Review 3, runs on Sreesuwin's laptop)
`docker-compose.yml`: service `ollama` (`ollama/ollama`, GPU reservation
`deploy.resources.reservations.devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]`,
volume `ollama:/root/.ollama`, port 11434) and service `app` (Python 3.11-slim, install
requirements, `streamlit run src/app/streamlit_app.py --server.port 8501`, env
`OLLAMA_URL=http://ollama:11434`, port 8501).

---

## 7. Sreesuwin S R – GenAI Advisor and LLM server

**Review 2:** Ollama shared on the network, real guardrail, chat on real modules.
**Review 3:** monthly report text, prompt tuning from eval failures, 3B fallback comparison.

### 7.1 Already working
`src/llm/tools.py` (7 tools, no tool takes `home_id`: the app fixes the home) and
`src/llm/advisor.py:chat` (tool loop up to 4 rounds, tool errors returned to the model,
one guardrail retry). Test: `python -m src.llm.advisor "Is a 5 kWh battery worth it for me?"`.

### 7.2 Guardrail – `src/llm/guardrail.py`
1. Numbers in reply: `re.findall(r"\d[\d,]*\.?\d*", reply)`, strip commas, to float.
2. Allowed set: every number found by walking all `tool_log[i]["output"]` values
   recursively + numbers in the question + integers 0–24 + years 2000–2035.
3. A number passes if `abs(x − a) <= max(0.05, 0.01 × |a|)` for some allowed `a`, or it equals
   `round(a)` or `round(a, 1)`.
4. Return `{"ok": not bad, "bad": bad, "ratio": round(passed / total, 2) if total else 1.0}`.

### 7.3 Prompts – `src/llm/prompts.py`
`SYSTEM_PROMPT` rules: never calculate; quote tool numbers exactly; ₹/kWh/kW; 3–6 sentences;
give saving + evidence; refuse wiring/safety, suggest a licensed electrician; savings are
estimates. Tune with Shree Vaishnavi's failure list. Add 1–2 few-shot examples if tool
selection is weak.

### 7.4 `monthly_report_text(home_id, month)`
Filter the home's data to the month, call `compute_kpis`, `detect_anomalies`,
`generate_recommendations`; put them in `REPORT_PROMPT`; run `verify_numbers` against them
(pass them as a tool_log); retry once if not ok.

### 7.5 Responsible AI (Review 3)
Local model (no data leaves the machine) · numeric guardrail · safety refusals ·
"estimates, not guarantees" line in replies and report · forecast ranges shown, not single values.

---

## 8. Shree Vaishnavi M – Evaluation, Documents, Pitch

**Review 2:** 25 test questions, `run_eval.py` running, screenshots, Document 2.
**Review 3:** metrics tables + charts, Document 3, pitch.

### 8.1 Test set – `eval/test_questions.json` (3 written, add 22 more)
Fields: `id, home_id, question, expected_tool (null if none), check_key, category`.

| Category | Count | Examples |
|---|---|---|
| utilization | 5 | What share of my solar do I export? When do I use most power? |
| inefficiency | 5 | Why is my bill so high at night? Any unusual days? |
| planning | 6 | Is a 5 kWh battery worth it? What if I add 2 kW of panels? What if I buy an EV and drive 40 km a day? Next 2 weeks' consumption? |
| recommendations | 4 | What should I change first? How can I cut my bill? |
| out_of_scope | 5 | How do I rewire my inverter? What will prices be next year? Write me a poem. |

### 8.2 Run – `python -m eval.run_eval [model]` → `eval/results_<model>.csv`
Columns: id, category, question, expected_tool, tools_called, correct_tool, verified,
unverified_numbers, latency_s, reply, rating_1_to_5 (team fills).

### 8.3 Metrics for Review 3
| Metric | How |
|---|---|
| Tool-selection accuracy | % correct_tool |
| Groundedness | % verified; mean guardrail ratio |
| Answer quality | mean team rating 1–5 |
| Safety | % of out_of_scope refused/redirected |
| Latency | median and 95th percentile |
| Forecast | MAPE Chronos vs baseline, 20 homes, last 30 days |
| Model comparison | qwen2.5:7b vs qwen2.5:3b |
| Guardrail effect | % verified before vs after retry |

### 8.4 Documents
Document 2 (Implementation and Prototype Development Report) and Document 3 (Final Project
Report) go in `../Documentations/02_Review_2/` and `03_Review_3/`. Keep an issues log
(problem, fix, owner) – it feeds "technical problem solving".

---

## 9. Schedule

| Session | Time | Goal |
|---|---|---|
| Mentoring 2 | 11:15 – 12:30 | Build core modules (timeline below) |
| Review 2 | 12:30 – 1:45 | Demo initial prototype; Document 2 |
| Mentoring 3 | 1:45 – 2:45 | Integration, prompt tuning, evaluation, Responsible AI |
| Review 3 | 2:45 – 4:30 | Final demo; Document 3, code, prototype |
| Online quiz | 4:30 – 5:00 | Individual, Blackboard |

Mentoring 2 by person:

| Time | Shashikiran | Sreesuwin | Divya | Shree Vaishnavi |
|---|---|---|---|---|
| 11:15 | Copy data, env, start build.py | Ollama on network, share IP | Env, run app on stubs | Test questions |
| 11:30 | Full pipeline | Real guardrail | Battery + load-shift sims | Test questions, issues log |
| **11:50 sync** | parquet files shared | guardrail done | sims pass tests | 25 questions done |
| 11:50 | Real KPIs, anomalies | Chat on real modules, prompt fixes | Rules R1, R2, R5; Overview | First eval run |
| **12:15 sync** | merge | merge | merge | screenshots, Document 2 |
| 12:15 | Baseline forecast | Fix tool errors | Recommendations tab | Finish Document 2 |

**Review 2 demo (2 min):** pick a home → Overview (≈ half the solar used at home) →
Recommendations (₹ saved) → Ask "Is a 5 kWh battery worth it for me?" → show the verified badge
and the tool call (numbers from code) → what comes in Mentoring 3.

---

## 10. Git workflow
- `main` must always pass `pytest -q`.
- Branch per person: `data`, `llm`, `app`, `eval`. Merge to `main` at the sync points.
- Commit small, commit often. Never commit `data/raw`, `data/processed`, `.venv`.
