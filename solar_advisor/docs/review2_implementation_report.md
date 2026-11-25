# Review 2 Implementation Report

**Project Title:** Intelligent Renewable Energy Optimization and Advisory Assistant (SolarSense)  
**Document Type:** Academic Project Report — Review 2 Template  
**Prepared By:** `[Team Member Names / Roll Numbers — TBD]`  
**Date:** `[Submission Date — TBD]`  

> [!NOTE]
> **Template Notice:** This document is a structured academic report template for Review 2. All dataset statistics, architectural specifics, model parameters, evaluation scores, and screenshots are intentionally marked as placeholders (`[TBD]`) to be filled only with verified outputs once the full application code and experimental results are integrated.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Project Objectives](#3-project-objectives)
4. [System Architecture](#4-system-architecture)
5. [Dataset Description](#5-dataset-description)
6. [Data Preprocessing and Cleaning](#6-data-preprocessing-and-cleaning)
7. [Feature Engineering](#7-feature-engineering)
8. [AI/ML Model Implementation](#8-aiml-model-implementation)
9. [Generative AI Integration](#9-generative-ai-integration)
10. [Analytics Engine](#10-analytics-engine)
11. [Recommendation Engine](#11-recommendation-engine)
12. [Application Prototype](#12-application-prototype)
13. [Testing and Validation](#13-testing-and-validation)
14. [Technical Challenges and Solutions](#14-technical-challenges-and-solutions)
15. [Screenshots of Implementation](#15-screenshots-of-implementation)
16. [Conclusion](#16-conclusion)

---

## 1. Abstract

`[TBD — Summarize the problem statement, SolarSense system design, forecasting and LLM components, evaluation approach, and verified Review 2 outcomes in 150–250 words after experiments are complete.]`

**Keywords:** Renewable Energy Optimization, Solar Analytics, Energy Forecasting, Generative AI, Guardrailed Tool Calling, SolarSense

---

## 2. Introduction

### 2.1 Background and Motivation
`[TBD — Describe residential solar/renewable energy adoption challenges, the gap between raw smart-meter telemetry and actionable homeowner guidance, and the motivation for SolarSense.]`

### 2.2 Problem Statement
`[TBD — State the exact technical and user-facing problem SolarSense addresses.]`

### 2.3 Scope of the Project
`[TBD — Define in-scope capabilities (energy analytics, forecasting, advisory assistant, safety guardrails) and out-of-scope boundaries.]`

---

## 3. Project Objectives

`[TBD — Confirm final project objectives with the team. Suggested structure below:]`

1. **Objective 1 (Data & Analytics):** `[TBD — e.g., Compute household renewable energy KPIs and detect consumption/generation inefficiencies.]`
2. **Objective 2 (Forecasting):** `[TBD — e.g., Forecast household energy profiles using Chronos-2 and benchmark against a seasonal baseline across 20 homes with a 30-day holdout.]`
3. **Objective 3 (Conversational Advisory & Guardrails):** `[TBD — e.g., Integrate tool-augmented LLM models (Qwen 2.5 7B and Qwen 2.5 3B) with grounding and safety guardrails.]`
4. **Objective 4 (Evaluation & Validation):** `[TBD — e.g., Benchmark accuracy, groundedness, answer quality, safety, and latency using the 25-question suite in eval/test_questions.json.]`

---

## 4. System Architecture

### 4.1 High-Level Architecture Overview
`[TBD — Describe the end-to-end data flow from raw telemetry -> preprocessing -> analytics/forecasting -> LLM + tools + guardrails -> user interface.]`

### 4.2 Architecture Diagram Placeholder

```text
[Placeholder: Insert verified System Architecture Diagram image or Mermaid diagram here]
```

### 4.3 Component Responsibilities

| Component / Module | Responsibility | Input | Output | Source Path (`TBD`) |
| :--- | :--- | :--- | :--- | :--- |
| Data Ingestion & Preprocessing | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Analytics & KPI Engine | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Forecasting Module (Chronos-2 / Baseline) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Recommendation Engine | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Generative AI & Tool Router | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Guardrails & Verification Layer | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| User Interface / Prototype | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Evaluation Suite | Runs 25-question benchmark | `eval/test_questions.json` | `results/results_*.csv` | [`eval/run_eval.py`](../eval/run_eval.py) |

---

## 5. Dataset Description

### 5.1 Data Sources
`[TBD — Document dataset origin, collection frequency, licensing/attribution, and number of homes.]`

### 5.2 Dataset Summary Statistics

| Statistic | Value (`TBD`) | Notes |
| :--- | :--- | :--- |
| Total Number of Homes | `[TBD]` | Forecast benchmark targets `20` homes |
| Time Range Covered | `[TBD]` | |
| Sampling Resolution / Interval | `[TBD]` | e.g., 15-min / hourly / daily |
| Total Records / Rows | `[TBD]` | |
| Number of Features / Columns | `[TBD]` | |
| Missing Value Rate (%) | `[TBD]` | |
| Holdout Window for Forecasting | Last `30` days | Coordinated with Shashikiran |

### 5.3 Schema Overview

| Column / Field Name (`TBD`) | Data Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

---

## 6. Data Preprocessing and Cleaning

### 6.1 Missing Values and Timestamp Alignment
`[TBD — Document how missing timestamps, null readings, and timezone/resampling alignments were handled.]`

### 6.2 Outlier Detection and Noise Filtering
`[TBD — Document checks for negative/impossible meter readings, sensor spikes, or duplicate rows.]`

### 6.3 Train–Test Split Strategy
`[TBD — Document exact split logic (e.g., holding out the last 30 days across 20 homes for forecast validation).]`

### 6.4 Cleaning Summary Table

| Preprocessing Step | Condition / Rule (`TBD`) | Records Affected (`TBD`) | Outcome |
| :--- | :--- | :--- | :--- |
| Duplicate Timestamp Removal | `[TBD]` | `[TBD]` | `[TBD]` |
| Missing Value Imputation | `[TBD]` | `[TBD]` | `[TBD]` |
| Outlier / Negative Value Handling | `[TBD]` | `[TBD]` | `[TBD]` |

---

## 7. Feature Engineering

### 7.1 Derived Temporal Features
`[TBD — Document time-based features such as hour of day, day of week, season, tariff period indicators, etc., once verified in code.]`

### 7.2 Energy & Lag Features
`[TBD — Document rolling averages, lag features, net load, or ratio features used by forecasting or analytics modules.]`

### 7.3 Feature Summary Table

| Feature Name (`TBD`) | Formula / Definition (`TBD`) | Target Module | Purpose |
| :--- | :--- | :--- | :--- |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

---

## 8. AI/ML Model Implementation

### 8.1 Baseline Forecasting Model (Seasonal Baseline)
`[TBD — Document the seasonal baseline formulation, lookback window, and assumptions.]`

### 8.2 Chronos-2 Forecasting Model
`[TBD — Document Chronos-2 configuration, context length, prediction horizon, and hardware/runtime setup.]`

### 8.3 Forecast Comparison Results (Placeholder)

*(See methodology in [`docs/evaluation_metrics.md`](./evaluation_metrics.md) Section 2.6 and Section 3.5; coordinate verified figures with **Shashikiran**.)*

| Model | Homes Evaluated | Holdout Window | Primary Error Metric (`TBD`) | Secondary Error Metric (`TBD`) | Improvement vs. Baseline (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Seasonal Baseline | 20 | Last 30 days | `[TBD]` | `[TBD]` | — |
| Chronos-2 | 20 | Last 30 days | `[TBD]` | `[TBD]` | `[TBD]` |

---

## 9. Generative AI Integration

### 9.1 Local LLM Configuration
- **Evaluated Models:** `qwen2.5:7b` and `qwen2.5:3b`
- **Inference Runtime & Parameters:** `[TBD — Document serving backend, temperature, context window, and hardware specs.]`

### 9.2 Tool Calling and Assistant Workflow
`[TBD — Document the exact registered tool names, schemas, and routing flow inside chat() once the main source code is integrated.]`

### 9.3 Guardrails and Grounding Verification
`[TBD — Document the actual input/output guardrails, out-of-scope refusal logic, and groundedness verification rules implemented in the codebase.]`

---

## 10. Analytics Engine

### 10.1 Key Performance Indicators (KPIs)
`[TBD — List exact KPI names, formulas, and units implemented in the Analytics Engine.]`

| KPI Key (`TBD`) | Display Name | Formula / Definition (`TBD`) | Units |
| :--- | :--- | :--- | :--- |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 10.2 Utilization and Inefficiency Detection
`[TBD — Describe how the engine evaluates solar utilization, grid dependency, and energy inefficiency patterns.]`

---

## 11. Recommendation Engine

### 11.1 Rule-Based / Optimization Logic
`[TBD — Describe how tariff windows, appliance shifting, battery scheduling, and solar planning recommendations are generated.]`

### 11.2 Advisory Output Structure
`[TBD — Document how analytical metrics and forecast outputs are translated into homeowner-friendly recommendations.]`

---

## 12. Application Prototype

### 12.1 User Interface Overview
`[TBD — Describe the frontend/prototype framework, pages/tabs, and user interaction flow.]`

### 12.2 Key Prototype Capabilities
- **Home Selection & Dashboard:** `[TBD]`
- **Interactive Q&A Assistant:** `[TBD]`
- **Visualizations & Reports:** `[TBD]`

---

## 13. Testing and Validation

### 13.1 Evaluation Framework Overview
The evaluation setup is defined in [`docs/evaluation_metrics.md`](./evaluation_metrics.md), executed via [`eval/run_eval.py`](../eval/run_eval.py), and tested against the `25`-question suite in [`eval/test_questions.json`](../eval/test_questions.json):
- `5` Utilization questions
- `5` Inefficiency questions
- `6` Planning questions
- `4` Recommendations questions
- `5` Out-of-scope / Safety questions

### 13.2 LLM Model Comparison Results (`qwen2.5:7b` vs. `qwen2.5:3b`)

| Metric | `qwen2.5:7b` | `qwen2.5:3b` | Difference (`7B - 3B`) |
| :--- | :---: | :---: | :---: |
| Tool Selection Accuracy (%) | `[TBD]` | `[TBD]` | `[TBD]` |
| Mean Guardrail Groundedness Ratio | `[TBD]` | `[TBD]` | `[TBD]` |
| Fully Verified Responses (%) | `[TBD]` | `[TBD]` | `[TBD]` |
| Answer Quality Mean (1–5) | `[TBD]` | `[TBD]` | `[TBD]` |
| Safety Rate (%) | `[TBD]` | `[TBD]` | `[TBD]` |
| Median Latency $P_{50}$ (s) | `[TBD]` | `[TBD]` | `[TBD]` |
| 95th Percentile Latency $P_{95}$ (s) | `[TBD]` | `[TBD]` | `[TBD]` |

### 13.3 Guardrail Ablation Results (With vs. Without Guardrail)

| Metric | With Guardrail | Without Guardrail | Change |
| :--- | :---: | :---: | :---: |
| Tool Selection Accuracy (%) | `[TBD]` | `[TBD]` | `[TBD]` |
| Mean Guardrail Groundedness Ratio | `[TBD]` | `[TBD]` | `[TBD]` |
| Fully Verified Responses (%) | `[TBD]` | `[TBD]` | `[TBD]` |
| Safety Rate (%) | `[TBD]` | `[TBD]` | `[TBD]` |
| Median Latency $P_{50}$ (s) | `[TBD]` | `[TBD]` | `[TBD]` |

### 13.4 Unit & Integration Checks
`[TBD — Document functional tests, JSON/CSV validation runs, and end-to-end smoke checks performed before Review 2 submission.]`

---

## 14. Technical Challenges and Solutions

All technical challenges, integration errors, data issues, evaluation bugs, and verified resolutions encountered during implementation are tracked systematically in [`docs/issues_log.md`](./issues_log.md).

### 14.1 Summary of Key Challenges (`[TBD]`)
`[TBD — Once docs/issues_log.md is populated with verified project issues, summarize the major challenges and implemented fixes here.]`

---

## 15. Screenshots of Implementation

> [!TIP]
> Save verified screenshot image files under `docs/screenshots/` (or a designated assets directory) and replace the placeholders below before final submission.

### 15.1 Application Home & KPI Dashboard
`[Placeholder: Insert screenshot of main SolarSense dashboard and KPI summary]`  
**Caption:** *Figure 15.1 — SolarSense Home Energy Dashboard (`[TBD]`).*

### 15.2 Interactive Advisory Chat & Tool Output
`[Placeholder: Insert screenshot of user query, tool invocation, and grounded assistant reply]`  
**Caption:** *Figure 15.2 — Conversational Assistant Answering a Utilization/Planning Query (`[TBD]`).*

### 15.3 Forecast Visualization (Chronos-2 vs. Baseline)
`[Placeholder: Insert screenshot of 30-day holdout forecast comparison plot]`  
**Caption:** *Figure 15.3 — Energy Forecast Comparison Across Holdout Window (`[TBD]`).*

### 15.4 Guardrail & Out-of-Scope Handling
`[Placeholder: Insert screenshot showing safe refusal/redirection on an out-of-scope or unsafe query]`  
**Caption:** *Figure 15.4 — Guardrail Handling of Out-of-Scope/Safety Query (`[TBD]`).*

### 15.5 Evaluation Script Output & CSV Results
`[Placeholder: Insert screenshot of eval/run_eval.py execution and generated CSV summary]`  
**Caption:** *Figure 15.5 — Evaluation Runner Execution (`[TBD]`).*

---

## 16. Conclusion

### 16.1 Summary of Review 2 Progress
`[TBD — Summarize implemented modules, validated metrics, and Review 2 milestones achieved.]`

### 16.2 Limitations Observed
`[TBD — Note verified system constraints, model trade-offs, or dataset limitations.]`

### 16.3 Future Work (Toward Final Review)
`[TBD — Outline planned improvements for Review 3 / final submission.]`
