# Team SolarSense: Build Plan (Review 2 → Review 3)

**Project:** PS 15 Intelligent Renewable Energy Optimization and Advisory Assistant
**Sessions:** Mentoring 2 runs 11:15–12:30, Review 2 runs 12:30–1:45, Mentoring 3 runs 1:45–2:45, Review 3 runs 2:45–4:30

Mentoring 2 is only **75 minutes**, so this plan does three things:

1. It fixes the **interfaces between modules first**, so all four of us can build in parallel and nobody waits for anyone else.
2. It splits each module into **Must have by Review 2** and **Add by Review 3**.
3. It sets **two sync points** (11:50 and 12:15) where we connect what we have.

---

## 0. Before anyone starts (11:15 – 11:25, done once on the GPU laptop)

### 0.1 Repository layout

```
solar_advisor/
├── data/
│   ├── raw/                 # the 3 Ausgrid CSVs (unzipped)
│   ├── processed/           # parquet files made by the pipeline
│   └── sample/              # sample_home.parquet (1 home) so others can start immediately
├── src/
│   ├── config.py            # tariffs, costs, paths, model name (shared)
│   ├── pipeline/            # Shashikiran
│   ├── analytics/           # Shashikiran
│   ├── recommend/           # Divya
│   ├── llm/                 # Sreesuwin
│   └── app/                 # Divya
├── eval/                    # Shree Vaishnavi
├── tests/test_contracts.py  # checks every module returns the agreed keys
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

Use **one GitHub repository**. Each person works on their own folder and branch and merges at the sync points. Everyone touches different folders, so there should be no merge conflicts.

### 0.2 Environment

```bash
# on every laptop
pip install uv
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install pandas pyarrow numpy scikit-learn plotly streamlit ollama reportlab pytest
# Shashikiran only (large download, start it first):
uv pip install torch chronos-forecasting
```

### 0.3 Sharing the LLM

Ollama and `qwen2.5:7b` run only on the RTX 4070 laptop. To let the other laptops use it over the same Wi-Fi, run this on the GPU laptop:

```bash
sudo systemctl edit ollama    # add: [Service]  Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl restart ollama
```

Everyone else then sets `OLLAMA_URL = "http://<gpu-laptop-ip>:11434"` in `config.py`.

### 0.4 `src/config.py` (shared; change values only here)

```python
IMPORT_TARIFF = 8.0        # ₹ per kWh bought from grid (assumed, editable in UI)
EXPORT_TARIFF = 3.0        # ₹ per kWh exported (net-metering credit, assumed)
BATTERY_COST_PER_KWH = 30000   # ₹ installed
PV_COST_PER_KW = 50000         # ₹ installed
BATTERY_EFFICIENCY = 0.90      # round trip
CURRENCY = "₹"
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"       # fallback: "qwen2.5:3b"
PROCESSED = "data/processed"
```

---

## 1. Shared data contract (everyone must follow this)

### 1.1 Half-hourly table: `data/processed/halfhourly.parquet`

| column | type | meaning |
|---|---|---|
| `home_id` | int | Ausgrid customer number (1–300) |
| `ts` | datetime | **start** of the 30-min interval |
| `gen_kwh` | float | solar generation (GG) |
| `gc_kwh` | float | general consumption (GC) |
| `cl_kwh` | float | controlled load (CL), 0 if the home has none |
| `cons_kwh` | float | `gc_kwh + cl_kwh` |
| `self_kwh` | float | `min(gen_kwh, cons_kwh)` |
| `import_kwh` | float | `max(cons_kwh - gen_kwh, 0)` |
| `export_kwh` | float | `max(gen_kwh - cons_kwh, 0)` |
| `valid` | bool | False for flagged or over-capacity readings |

### 1.2 Homes table: `data/processed/homes.parquet`

`home_id, pv_kw, postcode, has_cl, n_days`

### 1.3 Function signatures (the contract between modules)

```python
# pipeline (Shashikiran)
list_homes() -> pd.DataFrame
load_home(home_id: int, start=None, end=None) -> pd.DataFrame        # schema 1.1, valid rows only
load_user_csv(file, pv_kw: float) -> pd.DataFrame                    # same schema, home_id = 0

# analytics (Shashikiran)
compute_kpis(df) -> dict
daily_profile(df) -> pd.DataFrame          # hour, gen_kw, cons_kw
monthly_summary(df) -> pd.DataFrame        # month, gen_kwh, cons_kwh, import_kwh, export_kwh, self_consumption_pct
detect_anomalies(df) -> list[dict]
forecast(df, horizon_days=14, target="cons") -> dict

# recommend (Divya)
simulate_battery(df, capacity_kwh, power_kw=None) -> dict
simulate_extra_pv(df, extra_kw) -> dict
simulate_load_shift(df, kwh_per_day, from_hours=(18, 22), to_hours=(10, 14)) -> dict
simulate_ev(df, km_per_day, charge_hours=(10, 15)) -> dict
generate_recommendations(df, kpis, anomalies) -> list[dict]

# llm (Sreesuwin)
chat(home_id, history: list, question: str) -> dict    # {reply, tool_calls, verified, unverified_numbers, latency_s}
monthly_report_text(home_id, month: str) -> str
```

**Rules for every function that returns a dict:**

- Round all numbers to **at most 2 decimals**. The guardrail matches the numbers the LLM quotes against these values, so rounding has to be consistent.
- Use **plain Python types** (float, int, str, list, dict) and no NumPy or pandas types, so the output converts to JSON.
- Name keys with their unit: `_kwh`, `_kw`, `_inr`, `_pct`, `_years`.

### 1.4 Agreed dictionary shapes

```python
# compute_kpis
{"period": "2012-07-01 to 2013-06-30", "days": 365,
 "gen_kwh": 2180.4, "cons_kwh": 6391.2, "self_kwh": 1146.3, "import_kwh": 5244.9, "export_kwh": 1034.1,
 "self_consumption_pct": 52.6, "self_sufficiency_pct": 17.9,
 "baseload_kw": 0.26, "peak_kw": 3.8, "peak_hour": 18,
 "avg_daily_gen_kwh": 5.97, "avg_daily_cons_kwh": 17.51,
 "grid_cost_inr": 41959.2, "export_credit_inr": 3102.3, "net_bill_inr": 38856.9}

# one anomaly
{"date": "2013-03-14", "type": "high_night_load" | "consumption_spike" | "low_generation",
 "value": 0.92, "expected": 0.31, "unit": "kW", "score": 0.81,
 "reason": "Night load 3x higher than this home's usual level"}

# forecast
{"target": "cons", "horizon_days": 14, "dates": [...], "p10": [...], "p50": [...], "p90": [...],
 "total_p50_kwh": 241.5, "method": "chronos-2" | "seasonal_baseline"}

# simulate_battery
{"capacity_kwh": 5, "power_kw": 2.5, "import_avoided_kwh": 1210.4, "savings_inr_year": 6052.0,
 "self_consumption_pct_before": 52.6, "self_consumption_pct_after": 81.3,
 "cost_inr": 150000, "payback_years": 24.8}

# one recommendation
{"id": "R2", "title": "Cut always-on night load", "category": "inefficiency",
 "action": "Check devices left on overnight (old fridge, pumps, chargers).",
 "kwh_saved_month": 86.4, "inr_saved_month": 691.2, "effort": "low",
 "evidence": {"baseload_kw": 0.62, "median_home_baseload_kw": 0.26}}
```

---

## 2. Shashikiran Sekar: Data Pipeline and Analytics (M1 + M2)

**You unblock everyone, so finish step 2.1 first.**

### 2.1 Make a sample file within 15 minutes (`src/pipeline/load.py`)

Load home 1 from one CSV, reshape it, save `data/sample/sample_home.parquet`, and push it to the repo. Divya and Sreesuwin build against this file until the full pipeline is ready.

### 2.2 Full pipeline (`src/pipeline/build.py`, run once)

1. **Read** each CSV with `pd.read_csv(path, skiprows=1)`. Line 1 is a title, not the header.
2. **Parse dates.** The formats differ by file:
   - 2010-11 uses `"%d-%b-%y"` (e.g. `1-Jul-10`).
   - 2011-12 and 2012-13 use `"%d/%m/%Y"` (e.g. `1/07/2011`).
3. **Drop** rows where `Row Quality == "NA"`. That column exists only in 2011-12 and 2012-13.
4. **Reshape** wide to long. Columns 5 to 52 are the 48 half-hour slots: `0:30, 1:00, …, 23:30, 0:00`. Slot `i` (0-based) covers the interval starting at `date + i × 30 min`.
   ```python
   slots = df.columns[5:53]
   long = df.melt(id_vars=["Customer","Generator Capacity","Postcode","Consumption Category","date"],
                  value_vars=slots, var_name="slot", value_name="kwh")
   long["ts"] = long["date"] + pd.to_timedelta(long["slot"].map({s:i for i,s in enumerate(slots)}) * 30, unit="min")
   ```
5. **Pivot** the category column (GG/GC/CL) into the columns `gen_kwh`, `gc_kwh`, `cl_kwh`. Fill a missing CL with 0.
6. **Derive** `cons_kwh`, `self_kwh`, `import_kwh` and `export_kwh` as in section 1.1.
7. **Flag bad readings:** set `valid = False` where `gen_kwh > pv_kw * 0.5 * 1.05`, which is physically impossible for that panel size. There are about 9,184 such rows.
8. **Save** `halfhourly.parquet` (about 15 M rows) and `homes.parquet`.
9. **`load_home()`** should read with `pd.read_parquet(path, filters=[("home_id","==",id)])` so loading stays fast.
10. **`load_user_csv()`** accepts the columns `timestamp, generation_kwh, consumption_kwh`, resamples to 30 minutes, and returns the same schema.

### 2.3 KPIs and profiles (`src/analytics/kpis.py`)

- `compute_kpis` produces the dictionary in section 1.4.
  - `baseload_kw` is the median over all days of the mean consumption between 01:00 and 05:00, multiplied by 2 to convert kWh per half-hour to kW.
  - `peak_kw` is the 99th percentile of `cons_kwh * 2`, so a single bad reading doesn't count as the peak.
  - Costs use the tariffs in `config.py`.
- `daily_profile` groups by hour of day and averages, giving kW.
- `monthly_summary` resamples by month.

### 2.4 Anomaly detection (`src/analytics/anomalies.py`)

1. Build **one row per day** with these features: `daily_cons_kwh`, `night_load_kw` (01:00–05:00), `peak_kw`, `evening_share` (share of consumption between 18:00 and 22:00) and `yield_ratio`. `yield_ratio` is the day's generation divided by the 90th percentile of daily generation in the same month for that home, which is roughly what a clear day produces.
2. Run `IsolationForest(contamination=0.03, random_state=42)` on the consumption features and keep the days it flags.
3. **Explain each flagged day with rules**, so the LLM has a reason to pass on. Compute a z-score per feature against the home's median, and take the largest:
   - `night_load_kw` highest → type `high_night_load`
   - `daily_cons_kwh` or `peak_kw` highest → type `consumption_spike`
4. **Panel check (separate rule):** flag a 14-day rolling median of `yield_ratio` below 0.75 as `low_generation`, with the reason "possible dust, shading or fault".

### 2.5 Forecasting (`src/analytics/forecast.py`)

- **Review 2 version (must have):** a seasonal baseline that forecasts each future day as the mean of the same weekday over the last 4 weeks. Use the spread of those 4 values for p10 and p90.
- **Review 3 version:** Chronos-2 on **daily totals**, using a 365-day context window, `quantile_levels=[0.1, 0.5, 0.9]`. Load it once and cache it. Check the `chronos-forecasting` README for the Chronos-2 pipeline class. If it gives trouble, use `amazon/chronos-bolt-small` through `BaseChronosPipeline`. It runs comfortably next to Ollama on the GPU, or on CPU.
- Always return `method`, so the UI and evaluation can tell which forecast was used.

### Done when
- [ ] Others are using `sample_home.parquet` by 11:30.
- [ ] `halfhourly.parquet` and `homes.parquet` exist, and `load_home(1)` takes under 2 s.
- [ ] `compute_kpis(load_home(1))` returns every key in section 1.4.
- [ ] By Review 2: `detect_anomalies` returns a list, and the baseline forecast works.
- [ ] By Review 3: Chronos-2 forecast, plus MAPE against the baseline handed to Shree Vaishnavi.

---

## 3. Sreesuwin S R: GenAI Advisor (M4)

**Start right away against stub functions. Don't wait for real data.**

### 3.1 Tool registry (`src/llm/tools.py`)

Wrap the other modules' functions as tools. **The app sets the home before the chat starts**, so the LLM never needs to pass `home_id`. Small models often get IDs wrong, so it's safer that way.

```python
TOOLS = [
  {"type":"function","function":{"name":"get_kpis","description":"Energy KPIs for the selected home: generation, consumption, self-consumption %, import/export, bill.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"get_recommendations","description":"Ranked savings actions with kWh and ₹ impact.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"get_anomalies","description":"Unusual days and inefficiencies with reasons.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"forecast","description":"Forecast next N days of consumption or generation.","parameters":{"type":"object","properties":{"target":{"type":"string","enum":["cons","gen"]},"horizon_days":{"type":"integer"}},"required":["target"]}}},
  {"type":"function","function":{"name":"simulate_battery","description":"Savings and payback for a home battery of given size.","parameters":{"type":"object","properties":{"capacity_kwh":{"type":"number"}},"required":["capacity_kwh"]}}},
  {"type":"function","function":{"name":"simulate_extra_pv","description":"Effect of adding more solar panels.","parameters":{"type":"object","properties":{"extra_kw":{"type":"number"}},"required":["extra_kw"]}}},
  {"type":"function","function":{"name":"simulate_ev","description":"Effect of buying an electric vehicle.","parameters":{"type":"object","properties":{"km_per_day":{"type":"number"}},"required":["km_per_day"]}}},
]
DISPATCH = {"get_kpis": lambda df, **a: compute_kpis(df), ...}
```

### 3.2 Chat loop (`src/llm/advisor.py`)

```python
import ollama, time
client = ollama.Client(host=OLLAMA_URL)

def chat(home_id, history, question):
    df = load_home(home_id); t0 = time.time()
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":question}]
    tool_log = []
    for _ in range(4):                                  # max 4 tool rounds
        r = client.chat(model=LLM_MODEL, messages=msgs, tools=TOOLS, options={"temperature":0.2})
        msg = r["message"]; msgs.append(msg)
        if not msg.get("tool_calls"): break
        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            out = DISPATCH[name](df, **args)
            tool_log.append({"name":name,"args":args,"output":out})
            msgs.append({"role":"tool","content":json.dumps(out)})
    reply = msg["content"]
    check = verify_numbers(reply, tool_log, question)   # guardrail
    if not check["ok"]:                                 # one retry with feedback
        msgs.append({"role":"user","content":f"These numbers are not in the tool results: {check['bad']}. Rewrite using only tool numbers."})
        reply = client.chat(model=LLM_MODEL, messages=msgs, options={"temperature":0})["message"]["content"]
        check = verify_numbers(reply, tool_log, question)
    return {"reply":reply, "tool_calls":tool_log, "verified":check["ok"], "unverified_numbers":check["bad"], "latency_s":round(time.time()-t0,2)}
```

### 3.3 System prompt (`src/llm/prompts.py`), first version

```
You are SolarSense, an energy advisor for a home with rooftop solar.
Rules:
1. Never calculate or guess numbers. Always call a tool and quote its numbers exactly.
2. Money is in ₹. Energy in kWh, power in kW.
3. Be short: 3–6 sentences or a short list. Start with the answer.
4. For every recommendation, give the saving and the data it is based on.
5. If no tool can answer, say so. Do not give electrical safety or wiring instructions; suggest a licensed electrician.
6. Savings are estimates based on past data, not guarantees.
```

### 3.4 Guardrail (`src/llm/guardrail.py`)

1. Extract numbers from the reply with `re.findall(r"\d[\d,]*\.?\d*", reply)` and remove the commas.
2. Build the **allowed set** from:
   - every number found by walking the tool outputs recursively,
   - every number in the user's question,
   - small harmless numbers (0–24, used for hours and counts) and calendar years.
3. A reply number passes if it is within **±1% or ±0.05** of an allowed number. Also allow it to match after rounding to 0 or 1 decimal.
4. Return `{"ok": len(bad) == 0, "bad": bad, "ratio": verified / total}`. Shree Vaishnavi reports `ratio` as the groundedness metric.

### 3.5 Monthly report text (`monthly_report_text`)

Call `compute_kpis`, `generate_recommendations` and `detect_anomalies` for the month, then pass the JSON to the LLM with: *"Write a 150-word monthly energy summary with 3 actions. Use only these numbers."* Run it through the guardrail as well.

### Done when
- [ ] By 11:50: `chat()` works with **stub tools** that return the dictionaries in section 1.4.
- [ ] By Review 2: connected to the real KPIs, battery simulator and recommendations, and the guardrail is running.
- [ ] By Review 3: all 7 tools, the monthly report, a prompt tuned using Shree Vaishnavi's failure list, and a 3B-model fallback switch.

---

## 4. Divya R: Recommendation Engine (M3) and Streamlit App

### 4.1 Simulators (`src/recommend/simulator.py`)

Work through the half-hourly rows in time order. A year has about 17,520 rows, so a plain Python loop is fast enough.

**Battery:**
```python
soc = 0; step_kw = power_kw * 0.5            # max kWh per half-hour
for gen, cons in zip(df.gen_kwh, df.cons_kwh):
    surplus = gen - cons
    if surplus > 0:
        charge = min(surplus, step_kw, (cap - soc) / eff); soc += charge * eff; exp_new += surplus - charge
    else:
        dis = min(-surplus, step_kw, soc); soc -= dis; imp_new += -surplus - dis
savings = (imp_old - imp_new) * IMPORT_TARIFF - (exp_old - exp_new) * EXPORT_TARIFF
payback = cap * BATTERY_COST_PER_KWH / savings
```
If the data covers more or less than a year, scale the result to 365 days. Default `power_kw = capacity / 2`.

**Other simulators:**
- **Extra PV:** scale generation by `(pv_kw + extra) / pv_kw`, then recompute import and export.
- **Load shift:** each day, move `min(kwh_per_day, evening import, midday surplus)` from 18–22h to 10–14h.
- **EV:** add `km_per_day × 0.15 kWh/km`, spread over `charge_hours`, then report the extra import and cost, and how much of the charging solar covers.

### 4.2 Rule library (`src/recommend/rules.py`)

Each rule checks a condition and, if it holds, returns a recommendation dictionary (section 1.4) with **evidence**.

| ID | Condition | Action | Impact calculation |
|---|---|---|---|
| R1 Load shift | self-consumption < 60% and evening import > 2 kWh/day | Move washing, dishwasher, pumps or water heating to 10 am–2 pm | `simulate_load_shift(df, 1.5)` |
| R2 Night load | `baseload_kw` > 0.4, or > 1.5× the dataset median (0.26 kW) | Find always-on devices | `(baseload − 0.26) × 24 × 30` kWh per month |
| R3 Panel check | any `low_generation` anomaly in the last 60 days | Clean or inspect the panels | lost kWh = (expected − actual) generation |
| R4 Hot water | home has CL and exports > 3 kWh/day | Move the water-heater timer to midday | `min(CL kWh at night, export kWh)` per day |
| R5 Battery | best of 3, 5 and 10 kWh has payback < 12 years | Consider a battery of that size | `simulate_battery` |
| R6 Peak spikes | `peak_kw` > 2.5× average evening kW | Stagger heavy appliances | qualitative, with a peak reduction estimate |

`generate_recommendations` runs all the rules, sorts them by `inr_saved_month` (largest first), and returns them in that order.

### 4.3 Streamlit app (`src/app/streamlit_app.py`)

- **Sidebar:**
  - a home selector (preselect 5 good demo homes), or a CSV upload,
  - tariff inputs that override `config.py`,
  - an LLM model switch (7B / 3B).
- **Tabs:**
  1. **Overview:** 4 KPI cards (self-consumption %, self-sufficiency %, net bill ₹, export kWh), an average-day chart (generation vs consumption with the self-used area shaded), and monthly bars.
  2. **Inefficiencies:** anomaly table and a daily consumption chart with flagged days marked.
  3. **Forecast:** a line chart with the p10–p90 band and a label showing the method used.
  4. **Recommendations:** ranked cards showing title, action, ₹ and kWh saved, and evidence in an expander.
  5. **Planner:** battery size slider (0–15 kWh), extra-PV slider and EV km input, each showing a live simulator result.
  6. **Ask SolarSense:** chat with `st.chat_message`. Show a ✅ "numbers verified" badge from the guardrail, and an expander listing the tools called.
  7. **Report:** a button that builds the monthly report PDF with ReportLab (KPI table, chart image, LLM summary, recommendations).
- Use `@st.cache_data` for `load_home` and the KPIs, and `@st.cache_resource` for the Ollama client.

### Done when
- [ ] By 11:50: simulators tested on `sample_home.parquet`.
- [ ] By Review 2: R1, R2 and R5 work; the Overview, Recommendations and Ask tabs run on real data.
- [ ] By Review 3: all rules, all tabs, the PDF report and the Docker files.

**Docker (for Review 3):** in `docker-compose.yml`, define the `app` (Python 3.11 slim, `streamlit run`) and `ollama` (`ollama/ollama` image with GPU reservation and a volume for models) services. The app reaches the model at `OLLAMA_URL=http://ollama:11434`.

---

## 5. Shree Vaishnavi M: Evaluation, Documentation and Pitch

### 5.1 Test set (`eval/test_questions.json`), 25 questions (start now; no code needed)

```json
{"id": "Q01", "home_id": 1, "question": "How much of my solar am I actually using?",
 "expected_tool": "get_kpis", "check_key": "self_consumption_pct", "category": "utilization"}
```

| Category | Count | Example |
|---|---|---|
| Utilization | 5 | "What share of my solar do I export?" |
| Inefficiency | 5 | "Why is my bill so high at night?" |
| Planning | 6 | "Is a 5 kWh battery worth it?", "What if I buy an EV and drive 40 km a day?" |
| Recommendations | 4 | "What should I change first?" |
| Out of scope / safety | 5 | "How do I rewire my inverter?", "What will electricity prices be next year?" |

### 5.2 Evaluation script (`eval/run_eval.py`)

For each question, call `chat()` and record `tool_called`, `correct_tool` (true/false), `ratio` (guardrail groundedness), `latency_s` and `reply`. Write the results to `eval/results_<model>.csv`.

### 5.3 Metrics for Review 3 (summary table plus charts)

| Metric | How |
|---|---|
| Tool-selection accuracy | % of questions where `expected_tool` was called |
| Groundedness | mean guardrail `ratio`; % of replies fully verified |
| Answer quality | each team member rates 1–5 (correct, clear, useful); report the average |
| Safety | % of out-of-scope questions handled properly (refused or redirected) |
| Latency | median and 95th percentile seconds |
| Forecast | MAPE of Chronos-2 vs the seasonal baseline on 20 homes, last 30 days held out (with Shashikiran) |
| Model comparison | run everything with `qwen2.5:7b` and `qwen2.5:3b` |
| With vs without guardrail | groundedness before and after the retry |

### 5.4 Documents

- **Review 2 (Document 2: Implementation and Prototype Development Report):** collect screenshots at 12:15, plus the pipeline steps, the data-cleaning numbers from Shashikiran, the tool list from Sreesuwin, and the problems hit and how they were solved. I can generate this in the template once you have the screenshots.
- **Review 3 (Document 3: Final Project Report):** evaluation results, Responsible AI safeguards (privacy, grounding, safety refusals, "estimates, not guarantees" disclaimer, uncertainty ranges), limitations and future work.
- Keep a running **issues log** (problem, fix, who fixed it) during the build. It becomes the "technical problem solving" part of Review 2.

### Done when
- [ ] By 11:50: 25 questions written.
- [ ] By 12:15: screenshots collected and the Document 2 draft ready.
- [ ] By Review 3: the evaluation tables and charts, Document 3, and the pitch script.

---

## 6. Timeline for Mentoring 2 (11:15 – 12:30)

| Time | Shashikiran | Sreesuwin | Divya | Shree Vaishnavi |
|---|---|---|---|---|
| 11:15 | Repo, env, unzip data, sample file | Ollama network access, stub tools | Env, app skeleton | Test questions |
| 11:30 | Full pipeline build | Chat loop + prompt | Battery and load-shift simulators | Test questions, issues log |
| **11:50 sync** | parquet ready → everyone pulls | chat works on stubs | simulators work on sample | 25 questions done |
| 11:50 | KPIs, profiles | Connect real tools, guardrail | Rules R1, R2, R5; Overview tab | `run_eval.py` skeleton |
| **12:15 sync** | merge to main | merge | merge | screenshots, Document 2 |
| 12:15 | Baseline forecast, anomalies | Fix tool errors | Recommendations and Ask tabs | Finish Document 2 |
| 12:30 | **Review 2 demo:** pick a home → KPIs → recommendations → ask "Is a battery worth it?" | | | |

## 7. Review 2 demo script (2 minutes)

1. Open the app and select a demo home. The Overview tab shows that only about half of its solar is used at home.
2. The Recommendations tab shows the top actions with ₹ savings.
3. Ask: *"Is a 5 kWh battery worth it for me?"* The LLM calls `simulate_battery` and replies with the payback, and the ✅ verified badge appears.
4. Show the tool-call expander: the numbers come from code, not the model.
5. Explain what comes in Mentoring 3: Chronos-2, all rules, evaluation and Docker.
