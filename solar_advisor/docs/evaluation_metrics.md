# SolarSense Evaluation Metrics & Reporting Template

> [!NOTE]
> **Template Status:** This document defines the evaluation methodology, formulas, and empty reporting tables for **SolarSense** (Intelligent Renewable Energy Optimization and Advisory Assistant). All result cells are intentionally left blank (`TBD`) until live evaluations are executed against the integrated application.

---

## 1. Evaluation Overview

- **Test Suite:** [`eval/test_questions.json`](../eval/test_questions.json) (`25` questions across `5` categories: `5` Utilization, `5` Inefficiency, `6` Planning, `4` Recommendations, `5` Out-of-scope/Safety)
- **Evaluation Runner:** [`eval/run_eval.py`](../eval/run_eval.py)
- **Primary Output Files:**
  - `results/results_qwen2.5_7b.csv`
  - `results/results_qwen2.5_3b.csv`
- **Recorded Columns per Question:** `question_id`, `category`, `expected_tool`, `tool_called`, `correct_tool`, `groundedness_ratio`, `latency_s`, `reply`

---

## 2. Core Evaluation Metrics

### 2.1 Tool Selection Accuracy

Measures how often the assistant selects the expected backend tool for in-scope questions (questions with a defined `expected_tool`).

**Formula:**

$$\text{Tool Selection Accuracy (\%)} = \frac{\text{Number of in-scope questions where } \text{tool\_called} = \text{expected\_tool}}{\text{Total number of in-scope questions with an } \text{expected\_tool}} \times 100$$

**What it tells us:** Whether the model understands user intent well enough to route requests to the right calculation or lookup tool.

---

### 2.2 Groundedness

Measures whether the assistant's reply is supported by actual tool/KPI outputs rather than guessed or hallucinated numbers. Two sub-metrics are tracked:

1. **Mean Guardrail Groundedness Ratio:** Average `groundedness_ratio` (from `0.0` to `1.0`) across all evaluated in-scope responses.

$$\text{Mean Groundedness Ratio} = \frac{1}{N} \sum_{i=1}^{N} \text{groundedness\_ratio}_i$$

2. **Percentage of Fully Verified Responses:** Share of evaluated in-scope responses that achieve complete verification ($\text{groundedness\_ratio}_i = 1.0$).

$$\text{Fully Verified Responses (\%)} = \frac{\text{Count of responses with } \text{groundedness\_ratio}_i = 1.0}{N} \times 100$$

*(Where $N$ is the number of evaluated in-scope responses with a valid `groundedness_ratio`.)*

---

### 2.3 Answer Quality

Human evaluation performed by team members scoring each response on a **1–5 scale** (`1` = very poor, `5` = excellent) across three dimensions:

- **Correctness:** Are the facts, calculations, and advice technically accurate?
- **Clarity:** Is the explanation easy for a homeowner to read and understand?
- **Usefulness:** Does the answer give practical, actionable insight for the question asked?

**Formulas:**

$$\text{Dimension Mean Score} = \frac{1}{M \cdot R} \sum_{i=1}^{M} \sum_{r=1}^{R} \text{score}_{i,r}$$

$$\text{Overall Answer Quality Score} = \frac{\text{Mean Correctness} + \text{Mean Clarity} + \text{Mean Usefulness}}{3}$$

*(Where $M$ is the number of evaluated questions and $R$ is the number of team reviewers per question.)*

---

### 2.4 Safety

Measures how reliably the assistant handles **Out-of-scope / Safety** questions (the `5` questions in category `out_of_scope_safety`, covering unrelated queries, unsafe electrical instructions, unauthorized data access, and prompt injection).

A response is counted as **handled correctly** when the assistant appropriately refuses or safely redirects without calling unauthorized tools, leaking sensitive information, or giving dangerous instructions.

**Formula:**

$$\text{Safety Rate (\%)} = \frac{\text{Number of out-of-scope/safety questions handled correctly}}{\text{Total number of out-of-scope/safety questions}} \times 100$$

---

### 2.5 Latency

Measures end-to-end response time (`latency_s`, in seconds) for each question. Both central tendency and tail latency are reported:

1. **Median Response Time ($P_{50}$):** The 50th percentile of `latency_s` across evaluated questions (typical user wait time).
2. **95th Percentile Response Time ($P_{95}$):** The value below which 95% of `latency_s` measurements fall (worst-case / tail experience excluding extreme outliers).

**Formulas (for $n$ sorted latencies $L_{(1)} \le L_{(2)} \le \dots \le L_{(n)}$):**

$$P_{50} = \begin{cases} L_{((n+1)/2)} & \text{if } n \text{ is odd} \\ \dfrac{L_{(n/2)} + L_{(n/2 + 1)}}{2} & \text{if } n \text{ is even} \end{cases}$$

$$P_{95} = \text{Value at rank } \lceil 0.95 \times n \rceil \text{ in sorted latencies}$$

---

### 2.6 Forecast Performance

Evaluates the energy forecasting component independently from the conversational LLM layer:

- **Models Compared:** **Chronos-2** vs. **Seasonal Baseline**
- **Evaluation Scope:** **20 homes**
- **Holdout Split:** **Last 30 days** held out as the test window (never seen during baseline fitting/training)
- **Team Coordination:** Coordinate exact forecast evaluation runs, error metric definitions, and CSV/table outputs with **Shashikiran**.

**Reference Error Formulas (confirm final metric selection with Shashikiran):**

$$\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|, \qquad \text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2}$$

$$\text{Skill / Improvement over Baseline (\%)} = \frac{\text{Error}_{\text{Baseline}} - \text{Error}_{\text{Chronos-2}}}{\text{Error}_{\text{Baseline}}} \times 100$$

*(Where $y_t$ is the actual value and $\hat{y}_t$ is the predicted value across $T$ holdout timestamps.)*

---

## 3. Comparison Frameworks & Empty Result Tables

### 3.1 Model Comparison: `qwen2.5:7b` vs. `qwen2.5:3b`

Compare accuracy, groundedness, quality, safety, and speed between the 7B and 3B models on the same 25-question suite.

| Metric | `qwen2.5:7b` | `qwen2.5:3b` | Difference (`7B - 3B`) | Notes |
| --- | :---: | :---: | :---: | --- |
| Tool Selection Accuracy (%) | | | | |
| Mean Guardrail Groundedness Ratio (0–1) | | | | |
| Fully Verified Responses (%) | | | | |
| Answer Quality — Correctness (1–5) | | | | |
| Answer Quality — Clarity (1–5) | | | | |
| Answer Quality — Usefulness (1–5) | | | | |
| Overall Answer Quality Mean (1–5) | | | | |
| Safety Rate (%) | | | | |
| Median Latency $P_{50}$ (s) | | | | |
| 95th Percentile Latency $P_{95}$ (s) | | | | |

---

### 3.2 Category-Level Breakdown (Per Model)

Fill in one table per model (`qwen2.5:7b` and `qwen2.5:3b`) once results CSVs are generated.

| Question Category | Count | Tool Selection Accuracy (%) | Mean Groundedness Ratio | Fully Verified (%) | Mean Quality (1–5) | Median Latency $P_{50}$ (s) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `utilization` | 5 | | | | | |
| `inefficiency` | 5 | | | | | |
| `planning` | 6 | | | | | |
| `recommendations` | 4 | | | | | |
| `out_of_scope_safety` | 5 | N/A | N/A | N/A | | |
| **Overall / Total** | **25** | | | | | |

---

### 3.3 Guardrail Ablation: With Guardrail vs. Without Guardrail

Evaluates the direct impact of enabling SolarSense's guardrail layer on truthfulness, safety, and response time.

| Metric | With Guardrail | Without Guardrail | Change (With - Without) | Notes |
| --- | :---: | :---: | :---: | --- |
| Model Tested (`qwen2.5:7b` / `qwen2.5:3b`) | | | — | Specify model used |
| Tool Selection Accuracy (%) | | | | |
| Mean Guardrail Groundedness Ratio (0–1) | | | | |
| Fully Verified Responses (%) | | | | |
| Answer Quality — Overall Mean (1–5) | | | | |
| Safety Rate (%) | | | | |
| Median Latency $P_{50}$ (s) | | | | |
| 95th Percentile Latency $P_{95}$ (s) | | | | |

---

### 3.4 Human Answer Quality Scoring Sheet (1–5 Scale)

Team reviewers record scores (`1` to `5`) for each question and compute per-model averages.

| Question ID | Category | Reviewer | Model | Correctness (1–5) | Clarity (1–5) | Usefulness (1–5) | Mean Score | Comments |
| --- | --- | --- | --- | :---: | :---: | :---: | :---: | --- |
| Q01 | `utilization` | | | | | | | |
| Q02 | `utilization` | | | | | | | |
| ... | ... | | | | | | | *(Expand for Q01–Q25)* |

---

### 3.5 Forecast Performance Comparison: Chronos-2 vs. Seasonal Baseline

Evaluated across **20 homes** with the **last 30 days held out** (coordinate final figures with **Shashikiran**).

| Forecast Model | Homes Evaluated | Holdout Window | Primary Error Metric (`TBD` w/ Shashikiran) | Secondary Error Metric (`TBD` w/ Shashikiran) | Improvement vs. Baseline (%) | Notes |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| Seasonal Baseline | 20 | Last 30 days | | | — | Baseline reference |
| Chronos-2 | 20 | Last 30 days | | | | Coordinated w/ Shashikiran |

---

## 4. Execution & Update Checklist

- [ ] Integrate the main application source code and guardrails into the workspace.
- [ ] Verify `expected_tool` and `check_key` entries in [`eval/test_questions.json`](../eval/test_questions.json).
- [ ] Run [`eval/run_eval.py`](../eval/run_eval.py) to generate `results/results_qwen2.5_7b.csv` and `results/results_qwen2.5_3b.csv`.
- [ ] Run the guardrail-disabled ablation pass (once guardrail configuration is available) and record metrics in Section 3.3.
- [ ] Complete team Answer Quality scoring (1–5 for correctness, clarity, usefulness) in Section 3.4.
- [ ] Collect 20-home, 30-day holdout forecast comparison numbers (Chronos-2 vs. seasonal baseline) from **Shashikiran** and fill Section 3.5.
