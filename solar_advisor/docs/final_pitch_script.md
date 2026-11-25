# SolarSense — Final Project Presentation Pitch Script

**Project Title:** Intelligent Renewable Energy Optimization and Advisory Assistant (SolarSense)  
**Presenter Role:** Evaluation, Documentation, and Pitch  
**Audience:** University Project Review Panel  

> [!NOTE]
> **Before Presenting:** Replace every `[TBD]` placeholder with your verified numbers from `results/results_qwen2.5_7b.csv`, `results/results_qwen2.5_3b.csv`, and your team's forecast comparison with Shashikiran before your live defense. Do not read placeholder numbers as final results until the runs are completed.

---

## Part A: 2-Minute Short Pitch (Executive Summary)

*(Speak calmly at a steady pace — roughly 260–300 words.)*

> "Good morning/afternoon respected panel members. Today, I am presenting **SolarSense — our Intelligent Renewable Energy Optimization and Advisory Assistant**, focusing on the **Evaluation, Documentation, and Pitch** module.
>
> Many homeowners with solar panels struggle to understand raw electricity charts or know how to reduce grid dependence. SolarSense solves this by combining energy analytics, forecasting, and a guardrailed AI assistant that explains insights in simple, actionable language.
>
> For evaluation, we designed a structured **25-question benchmark** across five categories: **5 Utilization**, **5 Inefficiency**, **6 Planning**, **4 Recommendations**, and **5 Out-of-scope and Safety** questions. We selected 25 questions so the suite stays fast enough to run locally across models while thoroughly testing both real homeowner tasks and safety boundaries.
>
> We measure six core metrics: **Tool Selection Accuracy**, **Groundedness**, **Answer Quality** rated 1 to 5 by team members, **Safety**, **Latency** using median and 95th percentile response times, and **Forecast Performance**, where Chronos-2 is compared against a seasonal baseline across 20 homes with the last 30 days held out.
>
> On the language model side, our evaluation framework compares **Qwen 2.5 7B** against **Qwen 2.5 3B**, and also tests **with guardrails versus without guardrails** to verify that numbers come from real tools rather than guesses, and that unsafe requests are refused.
>
> *[Insert 1–2 sentences from live results once available: e.g., "Our final comparison showed [TBD]% tool accuracy and [TBD]s median latency for Qwen 2.5 7B versus [TBD]% and [TBD]s for Qwen 2.5 3B, while guardrails improved groundedness to [TBD]."]*
>
> Overall, SolarSense aims to make renewable energy advice accurate, transparent, and safe—while clearly stating that projections are data-driven estimates, not guarantees. Thank you, and I welcome your questions."

---

## Part B: 5-Minute Detailed Pitch (Structured by All 13 Topics)

*(Each section includes clear spoken lines you can deliver directly during a 5-minute university review.)*

### 1. Project Introduction
> "Respected faculty and panel members, our project is **SolarSense: Intelligent Renewable Energy Optimization and Advisory Assistant**. My responsibility in the team is **Evaluation, Documentation, and Pitch**—ensuring our system is measurable, transparent, and well-documented from Review 2 through the final report."

### 2. Problem Statement
> "Homeowners with solar panels and batteries often see raw kWh numbers on dashboards, but they still cannot answer practical questions like: *'How much of my solar am I actually using?'* or *'Which appliances should I shift to midday?'* At the same time, standard AI chatbots can hallucinate numbers or give unsafe electrical advice if they are not connected to real data and guardrails."

### 3. Project Objectives
> "To solve this, SolarSense has four main objectives:
> 1. Analyze household solar generation, consumption, and grid usage through clear KPIs.
> 2. Forecast future energy behavior using Chronos-2 benchmarked against a seasonal baseline.
> 3. Provide a conversational assistant powered by local Qwen 2.5 models connected to backend tools.
> 4. Enforce guardrails and rigorously evaluate accuracy, groundedness, quality, safety, and latency."

### 4. Evaluation Methodology
> "Rather than judging our AI assistant on one or two cherry-picked demos, we built a repeatable evaluation pipeline in `eval/run_eval.py` backed by `eval/test_questions.json` and documented in `docs/evaluation_metrics.md`. Every question is logged with the model's tool call, groundedness score, response latency, and full reply into CSV reports (`results/results_qwen2.5_7b.csv` and `results/results_qwen2.5_3b.csv`), alongside human quality scoring and forecast evaluation."

### 5. Why We Selected 25 Test Questions
> "We chose **25 carefully balanced questions** across five real-world categories:
> - **5 Utilization questions** to check basic solar and battery usage reporting.
> - **5 Inefficiency questions** to test whether the assistant identifies wasted energy or peak-tariff reliance.
> - **6 Planning questions** covering battery sizing, EV charging schedules, and outage backup scenarios.
> - **4 Recommendations questions** to evaluate practical homeowner action plans.
> - **5 Out-of-scope and Safety questions** covering smart-meter tampering, live high-voltage wiring, unauthorized neighbor data access, prompt injection, and unrelated trivia.
>
> Twenty-five questions give broad functional and safety coverage while remaining practical to evaluate end-to-end across multiple local models and human reviewers."

### 6. Evaluation Metrics
> "We evaluate SolarSense using six clear metrics:
> 1. **Tool Selection Accuracy:** Percentage of in-scope questions where the model called the expected backend tool.
> 2. **Groundedness:** Mean guardrail groundedness ratio and percentage of fully verified responses.
> 3. **Answer Quality:** Team members score responses from 1 to 5 on correctness, clarity, and usefulness.
> 4. **Safety:** Percentage of out-of-scope or unsafe questions handled with proper refusals.
> 5. **Latency:** Median ($P_{50}$) and 95th percentile ($P_{95}$) response times in seconds.
> 6. **Forecast Performance:** Chronos-2 compared against a seasonal baseline across 20 homes, holding out the last 30 days, coordinated with Shashikiran."

### 7. Qwen 2.5 7B vs Qwen 2.5 3B
> "We evaluate both **Qwen 2.5 7B** and **Qwen 2.5 3B** on the exact same 25 questions. This lets us study the practical trade-off between **reasoning/tool-calling accuracy** in the larger 7B model and **faster response speed and lower memory footprint** in the smaller 3B model."

### 8. Groundedness and Guardrails
> "In an energy advisory system, guessing a single number can mislead a homeowner. Our evaluation checks **with guardrails versus without guardrails** to measure how verification prevents unsupported claims, ensures responses stay grounded in tool outputs, and blocks unsafe or unauthorized requests."

### 9. Responsible AI and Safety
> "Responsible AI is built into SolarSense across five pillars:
> - **Data Privacy:** Queries are scoped to an authorized `home_id`, and local models avoid sending private home data to external servers.
> - **Grounded Responses:** Numeric claims are checked against real tool and KPI outputs.
> - **Safety Refusals:** Requests involving meter bypassing, live high-voltage DIY work, or prompt injection are refused safely.
> - **Uncertainty Ranges:** Forecasts and sizing estimates account for real-world weather and usage variability.
> - **Estimates, Not Guarantees Disclaimer:** We clearly inform users that savings and forecasts are data-driven estimates, not financial or engineering guarantees."

### 10. Model Evaluation Results (Placeholder)
> "*(Read only verified values after running live evaluations)*:
> - **Qwen 2.5 7B:** Tool Selection Accuracy = `[TBD]%`, Mean Groundedness = `[TBD]`, Fully Verified = `[TBD]%`, Mean Quality = `[TBD]/5`, Safety Rate = `[TBD]%`, Median Latency = `[TBD]s`, $P_{95}$ Latency = `[TBD]s`.
> - **Qwen 2.5 3B:** Tool Selection Accuracy = `[TBD]%`, Mean Groundedness = `[TBD]`, Fully Verified = `[TBD]%`, Mean Quality = `[TBD]/5`, Safety Rate = `[TBD]%`, Median Latency = `[TBD]s`, $P_{95}$ Latency = `[TBD]s`.
> - **Guardrail Ablation (With vs. Without):** Groundedness changed from `[TBD]` to `[TBD]`, and Safety Rate changed from `[TBD]%` to `[TBD]%`.
> - **Forecast Performance (20 Homes, 30-Day Holdout — w/ Shashikiran):** Seasonal Baseline error = `[TBD]` vs. Chronos-2 error = `[TBD]` (`[TBD]%` improvement)."

### 11. Limitations
> "We also recognize honest limitations of the current prototype:
> - Evaluation currently uses a 25-question benchmark and a 20-home, 30-day holdout forecast scope.
> - Local LLM performance depends on available CPU/GPU hardware, creating a speed-versus-accuracy trade-off between 3B and 7B models.
> - SolarSense provides advisory recommendations only—it does not directly control smart inverters or appliances."

### 12. Future Scope
> "For future work, SolarSense can be extended with real-time smart-meter and weather API integration, automatic utility tariff syncing, larger automated evaluation benchmarks, and uncertainty-aware multi-step appliance scheduling."

### 13. Conclusion
> "In conclusion, SolarSense demonstrates how energy analytics, forecasting, and local LLMs can work together safely when backed by structured evaluation and guardrails. Thank you for your time—I would be happy to answer any questions."

---

## Part C: Possible Panel Questions & Confident Student Answers

### Q1. Why did you choose only 25 test questions instead of hundreds?
**Suggested Answer:**
> "We chose 25 questions so the suite is focused, balanced, and practical to run completely across multiple local models (`qwen2.5:7b` and `qwen2.5:3b`) and guardrail modes, while also allowing team members to manually score every response from 1 to 5 on correctness, clarity, and usefulness. At the same time, the 5 categories cover all core tasks—utilization, inefficiency, planning, recommendations, and safety boundaries."

### Q2. Why compare Qwen 2.5 7B and Qwen 2.5 3B?
**Suggested Answer:**
> "In real-world local deployments, hardware resources matter. Comparing the 7B and 3B variants lets us quantify how much accuracy or groundedness we gain from a larger model versus how much latency and compute we save with a smaller model."

### Q3. How do you stop the LLM from making up fake electricity numbers?
**Suggested Answer:**
> "Instead of relying on the LLM's memory, the assistant is expected to call backend analytics/KPI tools and base its reply on those outputs. We track this directly through our **Tool Selection Accuracy** and **Groundedness** metrics, and compare results with and without guardrails."

### Q4. What happens if a user asks something dangerous, like how to tamper with a meter or rewire a panel?
**Suggested Answer:**
> "Our `out_of_scope_safety` test category specifically checks these cases—including smart-meter bypassing, live high-voltage rewiring, unauthorized access to another `home_id`, and prompt injection. The assistant is expected to refuse safely without executing tools or giving harmful instructions, and we report this as our **Safety** metric."

### Q5. Why do you include `P95` latency in addition to median latency?
**Suggested Answer:**
> "Median latency shows the typical experience for a homeowner, while 95th percentile (`P95`) latency reveals slow worst-case responses. Reporting both prevents occasional very slow queries from being hidden by an average."

### Q6. How is the forecasting model evaluated?
**Suggested Answer:**
> "Forecast performance is evaluated separately from the chat layer by comparing **Chronos-2** against a **seasonal baseline** across **20 homes**, holding out the **last 30 days** as unseen test data, in coordination with our teammate **Shashikiran**."

### Q7. Can homeowners treat SolarSense's savings numbers as guaranteed?
**Suggested Answer:**
> "No. Under our Responsible AI guidelines, we explicitly include an **'Estimates, Not Guarantees'** disclaimer and encourage uncertainty ranges, because real-world savings depend on weather, appliance habits, and tariff changes."

### Q8. What if the panel asks about the status of evaluation results right now (before live integration)?
**Suggested Answer (Honest & Professional):**
> "Right now, our evaluation suite (`eval/test_questions.json`), automated runner (`eval/run_eval.py`), and metrics/documentation templates (`docs/evaluation_metrics.md`, `docs/issues_log.md`, `docs/review2_implementation_report.md`, `docs/review3_final_report.md`) are ready. Once the main application code and `chat()` function are integrated into this workspace, we will execute the live runs for both Qwen 2.5 models, collect the 30-day holdout forecast metrics with Shashikiran, and fill in the verified tables without fabricating any numbers."
