# Review 3 Final Project Report

**Project Title:** Intelligent Renewable Energy Optimization and Advisory Assistant (SolarSense)  
**Document Type:** Academic Final Project Report — Review 3 Template  
**Prepared By:** `[Team Member Names / Roll Numbers — TBD]`  
**Guided By:** `[Faculty Guide / Mentor Name — TBD]`  
**Date:** `[Submission Date — TBD]`  

> [!NOTE]
> **Template Notice:** This document is a final academic report template for Review 3. All quantitative results, dataset statistics, system parameters, charts, screenshots, and bibliographic citations are intentionally marked as placeholders (`[TBD]`) and must be populated solely from verified code, experiments, and approved sources.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Project Objectives](#4-project-objectives)
5. [Proposed System](#5-proposed-system)
6. [System Architecture](#6-system-architecture)
7. [Dataset and Preprocessing](#7-dataset-and-preprocessing)
8. [AI/ML Models](#8-aiml-models)
9. [Generative AI Architecture](#9-generative-ai-architecture)
10. [Analytics Engine](#10-analytics-engine)
11. [Recommendation Engine](#11-recommendation-engine)
12. [Experimental Setup](#12-experimental-setup)
13. [Evaluation Methodology](#13-evaluation-methodology)
14. [Evaluation Results](#14-evaluation-results)
15. [Model Comparison (Qwen 2.5 7B vs Qwen 2.5 3B)](#15-model-comparison-qwen-25-7b-vs-qwen-25-3b)
16. [Guardrail Comparison (With vs Without Guardrail)](#16-guardrail-comparison-with-vs-without-guardrail)
17. [Forecast Comparison (Chronos-2 vs Seasonal Baseline)](#17-forecast-comparison-chronos-2-vs-seasonal-baseline)
18. [Responsible AI and Safety](#18-responsible-ai-and-safety)
19. [Limitations](#19-limitations)
20. [Future Enhancements](#20-future-enhancements)
21. [Conclusion](#21-conclusion)
22. [References](#22-references)

---

## 1. Abstract

`[TBD — Provide a concise academic abstract (200–300 words) summarizing the motivation, SolarSense architecture, forecasting models (Chronos-2 vs. Seasonal Baseline), LLM advisory models (Qwen 2.5 7B and Qwen 2.5 3B), guardrail design, and verified final evaluation outcomes.]`

**Keywords:** Renewable Energy Optimization, Solar Analytics, Time-Series Forecasting, Chronos-2, Local Large Language Models, Guardrailed Tool Calling, Responsible AI, SolarSense

---

## 2. Introduction

### 2.1 Context and Background
`[TBD — Introduce residential solar adoption, smart-meter monitoring, and the need for understandable, data-grounded advisory tools.]`

### 2.2 Motivation
`[TBD — Explain why raw energy dashboards are insufficient for non-technical homeowners and how combining forecasting, analytics, and conversational AI bridges this gap.]`

### 2.3 Organization of the Report
`[TBD — Briefly outline what each section of this final report covers.]`

---

## 3. Problem Statement

`[TBD — Formally define the core problem addressed by SolarSense: enabling accurate renewable energy analytics, short-to-medium-term forecasting, and reliable, non-hallucinated homeowner advisory across utilization, inefficiency, planning, recommendations, and safety boundaries.]`

---

## 4. Project Objectives

1. **Objective 1 — Energy Analytics & KPI Computation:** `[TBD — Summarize verified analytics goals.]`
2. **Objective 2 — Time-Series Forecasting:** `[TBD — Summarize Chronos-2 vs. Seasonal Baseline evaluation across 20 homes with a 30-day holdout.]`
3. **Objective 3 — Guardrailed Generative AI Advisory:** `[TBD — Summarize tool-augmented Qwen 2.5 7B / 3B integration and safety/grounding controls.]`
4. **Objective 4 — Empirical Evaluation & Responsible AI:** `[TBD — Summarize multi-metric benchmarking and responsible AI safeguards.]`

---

## 5. Proposed System

### 5.1 System Overview
`[TBD — Describe SolarSense as an integrated optimization and advisory system combining analytics, forecasting, recommendation logic, and a guardrailed conversational assistant.]`

### 5.2 Key Features
- **Utilization & Inefficiency Diagnostics:** `[TBD]`
- **Forecast-Driven Planning:** `[TBD]`
- **Actionable Recommendations:** `[TBD]`
- **Safety & Grounding Guardrails:** `[TBD]`

---

## 6. System Architecture

### 6.1 Architectural Design
`[TBD — Explain the modular pipeline connecting data storage, analytics/forecasting engines, LLM tool router, guardrails, and frontend interface.]`

### 6.2 Architecture Diagram Placeholder

```text
[Placeholder: Insert verified Final System Architecture Diagram here]
```

### 6.3 Module Summary

| Module | Purpose | Inputs (`TBD`) | Outputs (`TBD`) | Implementation File (`TBD`) |
| :--- | :--- | :--- | :--- | :--- |
| Data & Preprocessing Layer | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Analytics Engine | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Forecasting Engine | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Recommendation Engine | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Generative AI & Guardrails | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Evaluation Suite | Automated 25-question benchmark | `eval/test_questions.json` | `results/results_*.csv` | [`eval/run_eval.py`](../eval/run_eval.py) |

---

## 7. Dataset and Preprocessing

### 7.1 Dataset Characteristics
`[TBD — Document data source, number of homes, time span, sampling interval, and target variables.]`

### 7.2 Dataset Statistics Table (Placeholder)

| Parameter | Value (`TBD`) | Notes |
| :--- | :--- | :--- |
| Total Homes Available | `[TBD]` | `20` homes used in forecast benchmark |
| Date Range | `[TBD]` | |
| Resolution | `[TBD]` | |
| Total Observations | `[TBD]` | |
| Missing / Corrupted Records (%) | `[TBD]` | |
| Forecast Holdout Period | Last `30` days | Coordinated with Shashikiran |

### 7.3 Preprocessing and Feature Engineering Pipeline
`[TBD — Summarize verified data cleaning, missing-value handling, resampling, and engineered features.]`

---

## 8. AI/ML Models

### 8.1 Forecasting Models
- **Seasonal Baseline:** `[TBD — Document exact baseline method and seasonal period.]`
- **Chronos-2:** `[TBD — Document model configuration, inference parameters, and forecasting horizon.]`

### 8.2 Conversational LLM Models
- **`qwen2.5:7b`:** `[TBD — Document deployment details, quantization/parameters, and hardware environment.]`
- **`qwen2.5:3b`:** `[TBD — Document deployment details, quantization/parameters, and hardware environment.]`

---

## 9. Generative AI Architecture

### 9.1 Prompting and Tool-Calling Flow
`[TBD — Document system prompt structure, registered tool signatures, and how user queries map to backend tool calls.]`

### 9.2 Grounding and Guardrail Pipeline
`[TBD — Document pre-check safety filtering, tool output verification, groundedness scoring, and fallback/refusal behavior.]`

### 9.3 Generative AI Workflow Diagram Placeholder

```text
[Placeholder: Insert Tool-Calling and Guardrail Verification Flow Diagram here]
```

---

## 10. Analytics Engine

### 10.1 Computed KPIs
`[TBD — Document exact KPI keys, formulas, and units implemented in the final system.]`

| KPI Key (`TBD`) | Description | Formula (`TBD`) | Units |
| :--- | :--- | :--- | :--- |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 10.2 Diagnostic Logic
`[TBD — Explain how utilization metrics and inefficiency patterns are computed per home.]`

---

## 11. Recommendation Engine

### 11.1 Recommendation Generation Strategy
`[TBD — Document tariff optimization, load shifting, battery scheduling, and sizing advice logic.]`

### 11.2 Integration with Conversational Advisory
`[TBD — Explain how recommendations are surfaced cleanly to homeowners without unsupported claims.]`

---

## 12. Experimental Setup

### 12.1 Hardware and Software Environment

| Component | Specification (`TBD`) |
| :--- | :--- |
| CPU / GPU / RAM | `[TBD]` |
| Operating System | `[TBD]` |
| Python Version & Key Libraries | `[TBD]` |
| Local LLM Runtime | `[TBD]` |
| Forecast Evaluation Scope | `20` homes, last `30` days held out |
| LLM Evaluation Suite | `25` questions ([`eval/test_questions.json`](../eval/test_questions.json)) |

### 12.2 Test Suite Composition
- `5` Utilization questions (`Q01`–`Q05`)
- `5` Inefficiency questions (`Q06`–`Q10`)
- `6` Planning questions (`Q11`–`Q16`)
- `4` Recommendations questions (`Q17`–`Q20`)
- `5` Out-of-scope / Safety questions (`Q21`–`Q25`)

---

## 13. Evaluation Methodology

The evaluation framework, mathematical definitions, human-scoring rubrics, and comparison protocols follow [`docs/evaluation_metrics.md`](./evaluation_metrics.md).

### 13.1 Summary of Evaluation Metrics
1. **Tool Selection Accuracy:** Percentage of in-scope questions where `tool_called == expected_tool`.
2. **Groundedness:** Mean guardrail `groundedness_ratio` and percentage of fully verified responses (`groundedness_ratio = 1.0`).
3. **Answer Quality:** Team evaluation on a `1–5` scale across correctness, clarity, and usefulness.
4. **Safety:** Percentage of `out_of_scope_safety` questions handled appropriately without unsafe advice or unauthorized actions.
5. **Latency:** Median ($P_{50}$) and 95th percentile ($P_{95}$) response times in seconds.
6. **Forecast Performance:** Chronos-2 vs. Seasonal Baseline over `20` homes with the last `30` days held out (coordinated with **Shashikiran**).

---

## 14. Evaluation Results

### 14.1 Overall Benchmark Summary Table (Placeholder)

| Evaluation Dimension | `qwen2.5:7b` | `qwen2.5:3b` | Target / Notes (`TBD`) |
| :--- | :---: | :---: | :--- |
| Tool Selection Accuracy (%) | `[TBD]` | `[TBD]` | |
| Mean Guardrail Groundedness Ratio (0–1) | `[TBD]` | `[TBD]` | |
| Fully Verified Responses (%) | `[TBD]` | `[TBD]` | |
| Answer Quality — Correctness (1–5) | `[TBD]` | `[TBD]` | |
| Answer Quality — Clarity (1–5) | `[TBD]` | `[TBD]` | |
| Answer Quality — Usefulness (1–5) | `[TBD]` | `[TBD]` | |
| Overall Answer Quality Mean (1–5) | `[TBD]` | `[TBD]` | |
| Safety Rate (%) | `[TBD]` | `[TBD]` | |
| Median Latency $P_{50}$ (s) | `[TBD]` | `[TBD]` | |
| 95th Percentile Latency $P_{95}$ (s) | `[TBD]` | `[TBD]` | |

### 14.2 Category-Wise Results Table (Placeholder)

| Category | Count | `qwen2.5:7b` Accuracy (%) | `qwen2.5:3b` Accuracy (%) | `qwen2.5:7b` Quality (1–5) | `qwen2.5:3b` Quality (1–5) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `utilization` | 5 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `inefficiency` | 5 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `planning` | 6 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `recommendations` | 4 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `out_of_scope_safety` | 5 | N/A | N/A | `[TBD]` | `[TBD]` |
| **Total / Overall** | **25** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 14.3 Charts and Application Screenshots (Placeholders)

- **Figure 14.1 — Evaluation Metrics Overview Chart:** `[Placeholder: Insert bar/radar chart comparing metrics across categories]`
- **Figure 14.2 — Prototype UI Screenshot:** `[Placeholder: Insert verified screenshot of SolarSense UI]`

---

## 15. Model Comparison (Qwen 2.5 7B vs Qwen 2.5 3B)

### 15.1 Quantitative Comparison Table (Placeholder)

| Metric | `qwen2.5:7b` | `qwen2.5:3b` | Delta (`7B - 3B`) | Observed Trade-Off (`TBD`) |
| :--- | :---: | :---: | :---: | :--- |
| Tool Selection Accuracy (%) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Mean Groundedness Ratio | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Fully Verified Responses (%) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Overall Answer Quality (1–5) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Safety Rate (%) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Median Latency $P_{50}$ (s) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| 95th Percentile Latency $P_{95}$ (s) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 15.2 Comparative Analysis & Chart Placeholder
`[TBD — Discuss accuracy-vs-latency trade-offs observed in results/results_qwen2.5_7b.csv vs. results/results_qwen2.5_3b.csv.]`

- **Figure 15.1 — Qwen 2.5 7B vs. 3B Accuracy & Latency Chart:** `[Placeholder: Insert comparison chart]`

---

## 16. Guardrail Comparison (With vs Without Guardrail)

### 16.1 Ablation Results Table (Placeholder)

| Metric | With Guardrail | Without Guardrail | Change (`With - Without`) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Evaluated Model (`7B` / `3B`) | `[TBD]` | `[TBD]` | — | Specify model tested |
| Tool Selection Accuracy (%) | `[TBD]` | `[TBD]` | `[TBD]` | |
| Mean Groundedness Ratio | `[TBD]` | `[TBD]` | `[TBD]` | |
| Fully Verified Responses (%) | `[TBD]` | `[TBD]` | `[TBD]` | |
| Answer Quality Mean (1–5) | `[TBD]` | `[TBD]` | `[TBD]` | |
| Safety Rate (%) | `[TBD]` | `[TBD]` | `[TBD]` | |
| Median Latency $P_{50}$ (s) | `[TBD]` | `[TBD]` | `[TBD]` | |

### 16.2 Impact Analysis & Chart Placeholder
`[TBD — Analyze how enabling guardrails affected hallucination reduction, safety compliance, and latency overhead.]`

- **Figure 16.1 — With vs. Without Guardrail Comparison Chart:** `[Placeholder: Insert ablation chart]`

---

## 17. Forecast Comparison (Chronos-2 vs Seasonal Baseline)

*(Coordinated with **Shashikiran** across **20 homes** using the **last 30 days held out**.)*

### 17.1 Quantitative Forecast Table (Placeholder)

| Forecast Model | Homes Evaluated | Holdout Window | Primary Error Metric (`TBD`) | Secondary Error Metric (`TBD`) | Improvement vs. Baseline (%) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Seasonal Baseline | 20 | Last 30 days | `[TBD]` | `[TBD]` | — | Baseline reference |
| Chronos-2 | 20 | Last 30 days | `[TBD]` | `[TBD]` | `[TBD]` | Coordinated w/ Shashikiran |

### 17.2 Forecast Visualization & Analysis Placeholder
`[TBD — Summarize per-home error distribution, seasonal stability, and conditions where Chronos-2 outperforms or matches the baseline.]`

- **Figure 17.1 — 30-Day Holdout Forecast Plot (Chronos-2 vs. Seasonal Baseline):** `[Placeholder: Insert time-series forecast comparison chart]`
- **Figure 17.2 — Per-Home Forecast Error Distribution (20 Homes):** `[Placeholder: Insert box/bar plot across 20 homes]`

---

## 18. Responsible AI and Safety

SolarSense incorporates responsible AI design principles to ensure trustworthy, safe, and transparent guidance for homeowners:

### 18.1 Data Privacy
- **Home-Scoped Access:** Energy queries are scoped strictly to the authorized `home_id`, preventing cross-home data leakage (tested via unauthorized access cases in [`eval/test_questions.json`](../eval/test_questions.json)).
- **Local / Controlled Processing:** `[TBD — Document local LLM execution (`qwen2.5:7b` / `qwen2.5:3b`) and any data anonymization or PII-stripping practices verified in the final codebase.]`

### 18.2 Grounded Responses
- **Evidence-Backed Numbers:** Quantitative claims in advisory replies must trace back to verified tool outputs, analytics KPIs, or forecast results rather than unverified LLM parametric memory.
- **Groundedness Tracking:** Every evaluated response is audited via `groundedness_ratio` and verified response rates as defined in [`docs/evaluation_metrics.md`](./evaluation_metrics.md).

### 18.3 Safety Refusals
- **Hazardous & Illegal Requests:** Requests involving utility meter tampering, live high-voltage electrical rewiring, or unsafe DIY interventions must be firmly refused with safe redirection to licensed professionals.
- **Prompt Injection & Out-of-Scope Boundaries:** Attempts to override system instructions, extract internal schemas/secrets, or ask unrelated off-topic questions are blocked or redirected safely, measured through the `out_of_scope_safety` evaluation category.

### 18.4 Uncertainty Ranges
- **Probabilistic Forecasting & Variability:** Solar generation, household demand, tariff savings, and battery payback estimates depend on weather variability, occupancy changes, and equipment degradation.
- **Communicating Confidence Bounds:** `[TBD — Document how prediction intervals/uncertainty ranges (e.g., Chronos-2 quantile ranges or scenario bounds) are presented to users in the final UI and assistant replies.]`

### 18.5 Estimates, Not Guarantees Disclaimer
> [!IMPORTANT]
> **Advisory Disclaimer (Estimates, Not Guarantees):** All energy forecasts, bill-savings projections, payback estimates, and optimization recommendations provided by SolarSense are **data-driven estimates, not financial, engineering, or performance guarantees**. Actual energy generation, savings, and system behavior depend on weather conditions, utility tariff changes, appliance usage, and hardware performance. Qualified solar installers or licensed electricians should be consulted before making physical electrical modifications or major financial investments.

---

## 19. Limitations

`[TBD — Update with verified system limitations observed during final testing. Common dimensions to document once confirmed:]`

1. **Data Scope & Generalizability:** `[TBD — e.g., Dataset coverage, regional tariff assumptions, 20-home forecast scope.]`
2. **Model & Hardware Constraints:** `[TBD — e.g., Latency vs. reasoning capacity trade-off between qwen2.5:3b and qwen2.5:7b on local hardware.]`
3. **Evaluation Constraints:** `[TBD — e.g., 25-question benchmark sample size, subjective human Answer Quality variance.]`
4. **Real-Time / Control Boundaries:** `[TBD — e.g., Advisory-only recommendations without direct inverter/hardware actuation.]`

---

## 20. Future Enhancements

`[TBD — Document prioritized post-Review 3 enhancements aligned with observed limitations:]`

1. **Live Smart-Meter & Weather API Integration:** `[TBD]`
2. **Automated Tariff & Time-of-Use Sync:** `[TBD]`
3. **Expanded Evaluation Benchmark & Automated LLM-as-Judge Checks:** `[TBD]`
4. **Uncertainty-Aware Multi-Step Planning & Scheduling:** `[TBD]`

---

## 21. Conclusion

`[TBD — Synthesize final system achievements, verified insights from model/guardrail/forecast comparisons, and overall readiness of SolarSense upon completion of Review 3.]`

---

## 22. References

### Internal Project Documents
1. SolarSense Evaluation Methodology — [`docs/evaluation_metrics.md`](./evaluation_metrics.md)
2. SolarSense Technical Issues & Resolution Log — [`docs/issues_log.md`](./issues_log.md)
3. SolarSense Review 2 Implementation Report — [`docs/review2_implementation_report.md`](./review2_implementation_report.md)
4. SolarSense Evaluation Question Suite — [`eval/test_questions.json`](../eval/test_questions.json)
5. SolarSense Evaluation Runner — [`eval/run_eval.py`](../eval/run_eval.py)

### External References (`[TBD — Add only verified papers, libraries, and dataset citations used in the final implementation]`)
6. `[TBD — Author(s), "Title," Venue/Publisher, Year (e.g., Chronos-2 technical reference if used).]`
7. `[TBD — Author(s), "Title," Venue/Publisher, Year (e.g., Qwen 2.5 model reference if used).]`
8. `[TBD — Dataset citation and license reference.]`
