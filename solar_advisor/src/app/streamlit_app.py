"""SolarSense – Intelligent Renewable Energy Advisor (Streamlit Dashboard)."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

# Ensure repo root is importable when running `streamlit run src/app/streamlit_app.py`
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ollama
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import src.config as config
from src.app import adapters
from src.app.report import build_monthly_pdf_report
from src.recommend.rules import generate_recommendations
from src.recommend.simulator import (
    simulate_battery,
    simulate_ev,
    simulate_extra_pv,
    simulate_load_shift,
)

st.set_page_config(
    page_title="SolarSense – Intelligent Renewable Energy Advisor",
    page_icon="☀️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resource & data helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the AI model...")
def warm_up_llm(model_name: str) -> float:
    """Load the model into GPU memory once per app start, so the first question is fast."""
    try:
        from src.llm.advisor import warm_up
        return warm_up(model_name)
    except Exception:
        return 0.0


@st.cache_resource
def get_ollama_client(ollama_url: str) -> ollama.Client:
    return ollama.Client(host=ollama_url)


@st.cache_data(show_spinner=False)
def cached_list_homes() -> pd.DataFrame:
    return adapters.list_homes()


@st.cache_data(show_spinner="Loading household energy profile...")
def cached_load_home(home_id: int) -> pd.DataFrame:
    return adapters.load_home(int(home_id))


@st.cache_data(show_spinner=False)
def cached_compute_kpis(df: pd.DataFrame, import_tariff: float, export_tariff: float) -> dict:
    kpis = dict(adapters.compute_kpis(df))
    imp_kwh = float(kpis.get("import_kwh", 0.0))
    exp_kwh = float(kpis.get("export_kwh", 0.0))
    grid_cost = round(imp_kwh * import_tariff, 2)
    export_credit = round(exp_kwh * export_tariff, 2)
    kpis["grid_cost_inr"] = grid_cost
    kpis["export_credit_inr"] = export_credit
    kpis["net_bill_inr"] = round(grid_cost - export_credit, 2)
    return kpis


@st.cache_data(show_spinner=False)
def cached_daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    return adapters.daily_profile(df)


@st.cache_data(show_spinner=False)
def cached_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return adapters.monthly_summary(df)


@st.cache_data(show_spinner=False)
def cached_anomalies(df: pd.DataFrame) -> list[dict]:
    return adapters.detect_anomalies(df)


@st.cache_data(show_spinner=False)
def cached_forecast(df: pd.DataFrame, horizon_days: int, target: str) -> dict:
    return adapters.forecast(df, horizon_days=horizon_days, target=target)


@st.cache_data(show_spinner=False)
def cached_recommendations(
    df: pd.DataFrame,
    kpis: dict,
    anomalies: list[dict],
    import_tariff: float,
    export_tariff: float,
) -> list[dict]:
    return generate_recommendations(
        df,
        kpis,
        anomalies,
        import_tariff=import_tariff,
        export_tariff=export_tariff,
    )


def _apply_runtime_config(import_tariff: float, export_tariff: float, model_name: str) -> None:
    config.IMPORT_TARIFF = float(import_tariff)
    config.EXPORT_TARIFF = float(export_tariff)
    config.LLM_MODEL = model_name
    os.environ["LLM_MODEL"] = model_name


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("☀️ SolarSense – Intelligent Renewable Energy Advisor")
st.caption("PS15 · Analytics, Optimization Simulators & Grounded GenAI Advisory")

with st.sidebar:
    st.header("Household & Data Source")
    data_source = st.radio(
        "Data input mode",
        ["Demo Household", "Upload CSV"],
        horizontal=True,
    )

    homes_df = cached_list_homes()
    selected_home_id = 1
    selected_pv_kw = 3.78
    uploaded_file = None

    if data_source == "Demo Household":
        if homes_df.empty:
            st.warning("No demo households available. Upload a CSV to continue.")
        else:
            home_ids = homes_df["home_id"].astype(int).tolist()
            default_demo = [h for h in (1, 2, 3, 4, 5) if h in home_ids]
            display_ids = default_demo + [h for h in home_ids if h not in default_demo][:20]
            selected_home_id = st.selectbox(
                "Select Household ID",
                display_ids,
                format_func=lambda hid: (
                    f"Home {hid} ({homes_df.loc[homes_df['home_id'] == hid, 'pv_kw'].iloc[0]:.2f} kW PV)"
                    if "pv_kw" in homes_df.columns and not homes_df[homes_df["home_id"] == hid].empty
                    else f"Home {hid}"
                ),
            )
            row = homes_df[homes_df["home_id"] == selected_home_id]
            if not row.empty and "pv_kw" in row.columns:
                selected_pv_kw = float(row["pv_kw"].iloc[0])
            st.caption(f"Installed Solar Capacity: **{selected_pv_kw:.2f} kW**")
    else:
        uploaded_file = st.file_uploader("Upload half-hourly energy CSV", type=["csv"])
        selected_pv_kw = st.number_input(
            "Solar PV Capacity (kW)",
            min_value=0.1,
            max_value=50.0,
            value=3.5,
            step=0.5,
        )
        selected_home_id = 0

    st.divider()
    st.header("Tariffs & Assumptions")
    import_tariff = st.number_input(
        f"Grid Import Tariff ({config.CURRENCY}/kWh)",
        min_value=0.0,
        max_value=50.0,
        value=float(config.IMPORT_TARIFF),
        step=0.5,
    )
    export_tariff = st.number_input(
        f"Solar Export Credit ({config.CURRENCY}/kWh)",
        min_value=0.0,
        max_value=50.0,
        value=float(config.EXPORT_TARIFF),
        step=0.5,
    )

    st.divider()
    st.header("GenAI Model (Ollama)")
    llm_choice = st.radio(
        "LLM Model",
        ["qwen2.5:7b", "qwen2.5:3b"],
        index=0 if config.LLM_MODEL == "qwen2.5:7b" else 1,
        horizontal=True,
    )
    st.caption(f"Ollama Endpoint: `{config.OLLAMA_URL}`")

_apply_runtime_config(import_tariff, export_tariff, llm_choice)
_ = get_ollama_client(config.OLLAMA_URL)

if adapters.USING_MOCK_PIPELINE or adapters.USING_MOCK_ANALYTICS or adapters.USING_MOCK_LLM:
    missing_parts = []
    if adapters.USING_MOCK_PIPELINE:
        missing_parts.append("pipeline")
    if adapters.USING_MOCK_ANALYTICS:
        missing_parts.append("analytics")
    if adapters.USING_MOCK_LLM:
        missing_parts.append("LLM advisor")
    st.info(
        f"Running with temporary fallback adapters for: **{', '.join(missing_parts)}** "
        "(backed by real Ausgrid data; will switch automatically when team modules are added)."
    )

# ---------------------------------------------------------------------------
# Load household data
# ---------------------------------------------------------------------------

df: pd.DataFrame = pd.DataFrame()
home_label = f"Home {selected_home_id}"

try:
    if data_source == "Upload CSV":
        if uploaded_file is None:
            st.info("Upload a CSV file with `timestamp`, `generation_kwh`, and `consumption_kwh` columns in the sidebar.")
            st.stop()
        df = adapters.load_user_csv(uploaded_file, float(selected_pv_kw))
        home_label = f"Uploaded CSV ({selected_pv_kw:.2f} kW PV)"
    else:
        df = cached_load_home(selected_home_id)
except Exception as exc:
    st.error(f"Failed to load household data: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("No valid energy records were found for this household selection.")
    st.stop()

kpis = cached_compute_kpis(df, import_tariff, export_tariff)
daily_prof_df = cached_daily_profile(df)
monthly_df = cached_monthly_summary(df)
anomalies = cached_anomalies(df)
recommendations = cached_recommendations(df, kpis, anomalies, import_tariff, export_tariff)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Inefficiencies",
    "Forecast",
    "Recommendations",
    "Planner",
    "Ask SolarSense",
    "Monthly Energy Report",
])

# --- Tab 1: Overview ---
with tab1:
    st.subheader(f"Energy Overview — {home_label} ({kpis.get('period', 'N/A')})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solar Self-Consumption", f"{kpis.get('self_consumption_pct', 0.0):.1f}%")
    c2.metric("Energy Self-Sufficiency", f"{kpis.get('self_sufficiency_pct', 0.0):.1f}%")
    c3.metric("Estimated Net Bill", f"{config.CURRENCY}{kpis.get('net_bill_inr', 0.0):,.2f}")
    c4.metric("Solar Export Energy", f"{kpis.get('export_kwh', 0.0):,.1f} kWh")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Solar Generation", f"{kpis.get('gen_kwh', 0.0):,.1f} kWh")
    s2.metric("Total Consumption", f"{kpis.get('cons_kwh', 0.0):,.1f} kWh")
    s3.metric("Night Baseload", f"{kpis.get('baseload_kw', 0.0):.2f} kW")
    s4.metric("Peak Demand (99th pct)", f"{kpis.get('peak_kw', 0.0):.2f} kW")

    col_left, col_right = st.columns(2)
    with col_left:
        if not daily_prof_df.empty:
            self_used_kw = daily_prof_df[["gen_kw", "cons_kw"]].min(axis=1)
            fig_day = go.Figure()
            fig_day.add_trace(go.Scatter(
                x=daily_prof_df["hour"],
                y=self_used_kw,
                fill="tozeroy",
                fillcolor="rgba(34, 197, 94, 0.25)",
                line=dict(width=0),
                name="Self-Consumed Solar",
            ))
            fig_day.add_trace(go.Scatter(
                x=daily_prof_df["hour"],
                y=daily_prof_df["gen_kw"],
                mode="lines+markers",
                name="Solar Generation (kW)",
                line=dict(color="#F59E0B", width=2.5),
            ))
            fig_day.add_trace(go.Scatter(
                x=daily_prof_df["hour"],
                y=daily_prof_df["cons_kw"],
                mode="lines+markers",
                name="Consumption (kW)",
                line=dict(color="#2563EB", width=2.5),
            ))
            fig_day.update_layout(
                title="Daily Average Generation vs Consumption",
                xaxis_title="Hour of Day",
                yaxis_title="Average Power (kW)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_day, width="stretch")

    with col_right:
        if not monthly_df.empty:
            fig_month = px.bar(
                monthly_df,
                x="month",
                y=["gen_kwh", "cons_kwh"],
                barmode="group",
                title="Monthly Generation & Consumption",
                labels={"value": "Energy (kWh)", "month": "Month", "variable": "Channel"},
                color_discrete_map={"gen_kwh": "#F59E0B", "cons_kwh": "#2563EB"},
            )
            st.plotly_chart(fig_month, width="stretch")

    # Solar self-consumption breakdown
    self_kwh = float(kpis.get("self_kwh", 0.0))
    exp_kwh = float(kpis.get("export_kwh", 0.0))
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        fig_donut = px.pie(
            names=["Self-Consumed Solar", "Exported to Grid"],
            values=[max(self_kwh, 0.0), max(exp_kwh, 0.0)],
            hole=0.55,
            title="Solar Utilization Split",
            color_discrete_sequence=["#16A34A", "#F59E0B"],
        )
        st.plotly_chart(fig_donut, width="stretch")
    with sc2:
        if not monthly_df.empty:
            fig_sc = px.line(
                monthly_df,
                x="month",
                y="self_consumption_pct",
                markers=True,
                title="Monthly Solar Self-Consumption (%)",
                labels={"self_consumption_pct": "Self-Consumption (%)", "month": "Month"},
            )
            fig_sc.update_yaxes(range=[0, 105])
            st.plotly_chart(fig_sc, width="stretch")

# --- Tab 2: Inefficiencies ---
with tab2:
    st.subheader("Detected Energy Inefficiencies & Anomalies")
    anom_df = pd.DataFrame(anomalies)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Flagged Events", len(anomalies))
    a2.metric("High Night-Load Events", sum(1 for a in anomalies if a.get("type") == "high_night_load"))
    a3.metric("Consumption Spikes", sum(1 for a in anomalies if a.get("type") == "consumption_spike"))
    a4.metric("Low Solar Generation", sum(1 for a in anomalies if a.get("type") == "low_generation"))

    daily_cons = (
        df.assign(date=pd.to_datetime(df["ts"]).dt.date)
        .groupby("date", as_index=False)[["cons_kwh", "gen_kwh"]]
        .sum()
    )
    daily_cons["date_str"] = daily_cons["date"].astype(str)

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=daily_cons["date_str"],
        y=daily_cons["cons_kwh"],
        mode="lines",
        name="Daily Consumption (kWh)",
        line=dict(color="#2563EB", width=1.6),
    ))

    if not anom_df.empty:
        merged = daily_cons.merge(anom_df, left_on="date_str", right_on="date", how="inner")
        color_map = {
            "high_night_load": "#DC2626",
            "consumption_spike": "#EA580C",
            "low_generation": "#7C3AED",
        }
        for atype, grp in merged.groupby("type"):
            fig_anom.add_trace(go.Scatter(
                x=grp["date_str"],
                y=grp["cons_kwh"],
                mode="markers",
                name=f"Anomaly: {atype}",
                marker=dict(size=9, color=color_map.get(atype, "#DC2626"), symbol="diamond"),
                text=grp["reason"],
                hovertemplate="%{x}<br>Consumption: %{y:.2f} kWh<br>%{text}<extra></extra>",
            ))
    fig_anom.update_layout(
        title="Daily Consumption with Anomalous Dates Highlighted",
        xaxis_title="Date",
        yaxis_title="Consumption (kWh/day)",
    )
    st.plotly_chart(fig_anom, width="stretch")

    if anom_df.empty:
        st.success("No unusual consumption or low-generation anomalies were detected.")
    else:
        type_filter = st.multiselect(
            "Filter by anomaly type",
            options=sorted(anom_df["type"].unique().tolist()),
            default=sorted(anom_df["type"].unique().tolist()),
        )
        filtered = anom_df[anom_df["type"].isin(type_filter)]
        st.dataframe(filtered, width="stretch", hide_index=True)

# --- Tab 3: Forecast ---
with tab3:
    st.subheader("Energy Generation & Consumption Forecast")
    fc1, fc2 = st.columns(2)
    with fc1:
        fc_target = st.radio(
            "Forecast Target",
            ["cons", "gen"],
            format_func=lambda t: "Consumption (kWh/day)" if t == "cons" else "Solar Generation (kWh/day)",
            horizontal=True,
        )
    with fc2:
        horizon_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=30, value=14)

    fc_res = cached_forecast(df, horizon_days=horizon_days, target=fc_target)
    f1, f2, f3 = st.columns(3)
    f1.metric("Forecasted Total (p50)", f"{fc_res.get('total_p50_kwh', 0.0):,.2f} kWh")
    avg_p50 = fc_res["total_p50_kwh"] / max(len(fc_res.get("dates", [])), 1)
    f2.metric("Daily Average (p50)", f"{avg_p50:.2f} kWh/day")
    f3.metric("Forecast Method", str(fc_res.get("method", "unknown")))

    if fc_res.get("dates"):
        hist_col = "gen_kwh" if fc_target == "gen" else "cons_kwh"
        hist_daily = (
            df.assign(date=pd.to_datetime(df["ts"]).dt.date)
            .groupby("date", as_index=False)[hist_col]
            .sum()
            .tail(30)
        )
        hist_daily["date_str"] = hist_daily["date"].astype(str)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist_daily["date_str"],
            y=hist_daily[hist_col],
            mode="lines+markers",
            name="Recent Historical",
            line=dict(color="#475569", width=2),
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_res["dates"] + fc_res["dates"][::-1],
            y=fc_res["p90"] + fc_res["p10"][::-1],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="p10–p90 Uncertainty Band",
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_res["dates"],
            y=fc_res["p50"],
            mode="lines+markers",
            name="Median Forecast (p50)",
            line=dict(color="#2563EB", width=2.5),
        ))
        fig_fc.update_layout(
            title=f"{horizon_days}-Day {'Consumption' if fc_target == 'cons' else 'Solar Generation'} Forecast",
            xaxis_title="Date",
            yaxis_title="Daily Energy (kWh)",
        )
        st.plotly_chart(fig_fc, width="stretch")
    else:
        st.info("Not enough historical data to generate a forecast.")

# --- Tab 4: Recommendations ---
with tab4:
    st.subheader("Ranked Optimization Recommendations")
    if not recommendations:
        st.info("No specific optimization rules were triggered for this household.")
    else:
        total_monthly_inr = sum(r["inr_saved_month"] for r in recommendations)
        total_monthly_kwh = sum(r["kwh_saved_month"] for r in recommendations)
        st.caption(
            f"{len(recommendations)} recommendation(s) found · Combined potential: "
            f"**{total_monthly_kwh:,.1f} kWh/month** ({config.CURRENCY}{total_monthly_inr:,.2f}/month) — estimates only."
        )
        for rec in recommendations:
            with st.container(border=True):
                h_col, m_col1, m_col2 = st.columns([3, 1, 1])
                with h_col:
                    st.markdown(f"### {rec['id']} · {rec['title']}")
                    st.caption(f"Category: `{rec['category']}` · Effort: **{rec['effort'].title()}**")
                    st.write(rec["action"])
                with m_col1:
                    st.metric("Monthly kWh Saved", f"{rec['kwh_saved_month']:,.2f} kWh")
                with m_col2:
                    st.metric("Monthly INR Saved", f"{config.CURRENCY}{rec['inr_saved_month']:,.2f}")
                with st.expander("Supporting Evidence"):
                    st.json(rec["evidence"])

# --- Tab 5: Planner ---
with tab5:
    st.subheader("Interactive What-If Optimization Planner")
    p1, p2, p3 = st.columns(3)
    with p1:
        bat_kwh = st.slider("Battery Capacity (kWh)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
    with p2:
        extra_pv_kw = st.slider("Additional Solar PV (kW)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    with p3:
        ev_km = st.number_input("EV Daily Driving Distance (km/day)", min_value=0.0, max_value=200.0, value=40.0, step=5.0)

    shift_kwh = st.slider("Daily Load Shifted to Midday (kWh/day)", min_value=0.0, max_value=6.0, value=1.5, step=0.5)

    bat_res = simulate_battery(df, capacity_kwh=bat_kwh, import_tariff=import_tariff, export_tariff=export_tariff)
    pv_res = simulate_extra_pv(
        df,
        extra_kw=extra_pv_kw,
        pv_kw=selected_pv_kw,
        import_tariff=import_tariff,
        export_tariff=export_tariff,
    )
    ev_res = simulate_ev(df, km_per_day=ev_km, import_tariff=import_tariff, export_tariff=export_tariff)
    shift_res = simulate_load_shift(df, kwh_per_day=shift_kwh, import_tariff=import_tariff, export_tariff=export_tariff)

    col_b, col_p = st.columns(2)
    with col_b:
        with st.container(border=True):
            st.markdown("#### 🔋 Battery Storage Simulator")
            b1, b2, b3 = st.columns(3)
            b1.metric("Annual Savings", f"{config.CURRENCY}{bat_res['savings_inr_year']:,.2f}")
            b2.metric("Grid Import Avoided", f"{bat_res['import_avoided_kwh']:,.1f} kWh/yr")
            b3.metric("Estimated Payback", f"{bat_res['payback_years']} yrs")
            st.caption(
                f"Self-consumption: {bat_res['self_consumption_pct_before']}% → **{bat_res['self_consumption_pct_after']}%** · "
                f"Installed Cost: {config.CURRENCY}{bat_res['cost_inr']:,.0f} · Power: {bat_res['power_kw']} kW"
            )

    with col_p:
        with st.container(border=True):
            st.markdown("#### ☀️ Extra Solar PV Simulator")
            e1, e2, e3 = st.columns(3)
            e1.metric("Annual Savings", f"{config.CURRENCY}{pv_res['savings_inr_year']:,.2f}")
            e2.metric("Extra Generation", f"{pv_res['extra_gen_kwh']:,.1f} kWh/yr")
            e3.metric("Estimated Payback", f"{pv_res['payback_years']} yrs")
            st.caption(
                f"Self-consumption: {pv_res['self_consumption_pct_before']}% → **{pv_res['self_consumption_pct_after']}%** "
                f"({pv_res['self_consumption_change_pct']:+.2f}%) · Import avoided: {pv_res['import_avoided_kwh']:,.1f} kWh/yr"
            )

    col_e, col_s = st.columns(2)
    with col_e:
        with st.container(border=True):
            st.markdown("#### 🚗 EV Daytime Charging Simulator")
            v1, v2, v3 = st.columns(3)
            v1.metric("Solar Coverage", f"{ev_res['solar_coverage_pct']}%")
            v2.metric("Grid Charging Needed", f"{ev_res['ev_from_grid_kwh']:,.1f} kWh/yr")
            v3.metric("Est. Additional Cost", f"{config.CURRENCY}{ev_res['additional_cost_inr_month']:,.2f}/mo")
            st.caption(
                f"Total EV demand: {ev_res['ev_demand_kwh']:,.1f} kWh/yr ({ev_res['ev_demand_day_kwh']} kWh/day) · "
                f"From solar: {ev_res['ev_from_solar_kwh']:,.1f} kWh/yr"
            )

    with col_s:
        with st.container(border=True):
            st.markdown("#### ⏱️ Appliance Load-Shifting Simulator")
            l1, l2, l3 = st.columns(3)
            l1.metric("Monthly Savings", f"{config.CURRENCY}{shift_res['savings_inr_month']:,.2f}/mo")
            l2.metric("Grid Import Avoided", f"{shift_res['import_avoided_kwh']:,.1f} kWh/yr")
            l3.metric("Actual Shifted", f"{shift_res['avg_daily_shifted_kwh']} kWh/day")
            st.caption(
                f"Self-consumption: {shift_res['self_consumption_pct_before']}% → **{shift_res['self_consumption_pct_after']}%** · "
                f"Shifted from 18:00–22:00 to 10:00–14:00"
            )

# --- Tab 6: Ask SolarSense ---
with tab6:
    warm_up_llm(llm_choice)
    st.subheader("Ask SolarSense (Grounded Energy Advisor)")
    st.caption(
        f"Active model: `{llm_choice}` · Every numerical answer is checked against computed module outputs."
    )
    if selected_home_id == 0:
        st.info("Note: Sreesuwin's `chat(home_id, ...)` takes a persisted household ID; uploaded CSVs (`home_id=0`) work in fallback mode or once saved to the pipeline.")

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    history = st.session_state.chat_histories.setdefault(selected_home_id, [])

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("verified"):
                    st.success("✅ Numbers verified against module outputs")
                elif msg.get("unverified_numbers"):
                    st.warning(f"⚠️ Unverified numbers detected: {msg['unverified_numbers']}")
                if msg.get("tool_calls"):
                    with st.expander(f"Tool calls ({len(msg['tool_calls'])}) · {msg.get('latency_s', 0)}s"):
                        st.json(msg["tool_calls"])

    user_q = st.chat_input("Ask e.g. 'Is a 5 kWh battery worth it for me?' or 'What should I change first?'")
    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)

        api_history = [{"role": m["role"], "content": m["content"]} for m in history]
        try:
            with st.spinner("Consulting SolarSense advisor..."):
                chat_params = inspect.signature(adapters.chat).parameters
                chat_kwargs = {}
                if "model" in chat_params:
                    chat_kwargs["model"] = llm_choice
                if "df" in chat_params and selected_home_id == 0:
                    chat_kwargs["df"] = df
                resp = adapters.chat(selected_home_id, api_history, user_q, **chat_kwargs)
        except Exception as exc:
            resp = {
                "reply": f"Unable to complete LLM request: {exc}",
                "tool_calls": [],
                "verified": False,
                "unverified_numbers": [],
                "latency_s": 0.0,
            }

        history.append({"role": "user", "content": user_q})
        history.append({
            "role": "assistant",
            "content": resp.get("reply", ""),
            "tool_calls": resp.get("tool_calls", []),
            "verified": bool(resp.get("verified", False)),
            "unverified_numbers": resp.get("unverified_numbers", []),
            "latency_s": resp.get("latency_s", 0.0),
        })
        st.rerun()

# --- Tab 7: Monthly Energy Report ---
with tab7:
    st.subheader("Monthly Energy Report (PDF)")
    avail_months = monthly_df["month"].tolist() if not monthly_df.empty else ["Full Period"]
    chosen_month = st.selectbox("Select Month for Report Summary", avail_months[::-1])

    if st.button("Generate Monthly PDF Report", type="primary"):
        try:
            with st.spinner("Generating AI summary and building ReportLab PDF..."):
                ai_text = adapters.monthly_report_text(selected_home_id, str(chosen_month))
                pdf_bytes = build_monthly_pdf_report(
                    home_label=home_label,
                    month=str(chosen_month),
                    kpis=kpis,
                    monthly_df=monthly_df,
                    anomalies=anomalies,
                    recommendations=recommendations,
                    ai_summary=ai_text,
                    import_tariff=import_tariff,
                    export_tariff=export_tariff,
                )
            st.session_state["last_pdf_bytes"] = pdf_bytes
            st.session_state["last_pdf_summary"] = ai_text
            st.session_state["last_pdf_month"] = str(chosen_month)
        except Exception as exc:
            st.error(f"Failed to generate PDF report: {exc}")

    if "last_pdf_bytes" in st.session_state:
        st.markdown("#### Report Preview")
        st.info(st.session_state.get("last_pdf_summary", ""))
        st.download_button(
            label="⬇️ Download Monthly Energy Report (PDF)",
            data=st.session_state["last_pdf_bytes"],
            file_name=f"solarsense_report_home_{selected_home_id}_{st.session_state.get('last_pdf_month', 'monthly')}.pdf",
            mime="application/pdf",
        )
