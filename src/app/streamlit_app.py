"""SolarSense Streamlit app.   OWNER: Divya

Run from the repo root:  streamlit run src/app/streamlit_app.py
Skeleton: Overview + Ask tabs work on stubs. Add the remaining tabs from README 6.3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.analytics.kpis import compute_kpis, daily_profile  # noqa: E402
from src.pipeline.load import list_homes, load_home  # noqa: E402

st.set_page_config(page_title="SolarSense", layout="wide")


@st.cache_data
def _home(home_id):
    return load_home(home_id)


with st.sidebar:
    st.header("SolarSense")
    homes = list_homes()
    home_id = st.selectbox("Home", homes["home_id"].tolist())
    model = st.radio("LLM", [config.LLM_MODEL, config.LLM_FALLBACK_MODEL], horizontal=True)

df = _home(home_id)
kpis = compute_kpis(df)

tabs = st.tabs(["Overview", "Inefficiencies", "Forecast", "Recommendations", "Planner", "Ask SolarSense", "Report"])

with tabs[0]:
    c = st.columns(4)
    c[0].metric("Solar used at home", f"{kpis['self_consumption_pct']}%")
    c[1].metric("Demand met by solar", f"{kpis['self_sufficiency_pct']}%")
    c[2].metric("Net bill", f"{config.CURRENCY}{kpis['net_bill_inr']:,.0f}")
    c[3].metric("Exported", f"{kpis['export_kwh']:,.0f} kWh")
    prof = daily_profile(df)
    fig = go.Figure()
    fig.add_scatter(x=prof.hour, y=prof.gen_kw, name="Solar generation", line=dict(color="#CA5B1D", width=3))
    fig.add_scatter(x=prof.hour, y=prof.cons_kw, name="Consumption", line=dict(color="#143670", width=3))
    fig.update_layout(title="Average day (kW)", xaxis_title="Hour of day", height=380)
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    from src.llm.advisor import chat
    if "history" not in st.session_state:
        st.session_state.history = []
    for m in st.session_state.history:
        st.chat_message(m["role"]).write(m["content"])
    if q := st.chat_input("Ask about your solar and energy use"):
        st.chat_message("user").write(q)
        with st.spinner("Thinking..."):
            res = chat(home_id, st.session_state.history, q, model=model)
        with st.chat_message("assistant"):
            st.write(res["reply"])
            st.caption(("✅ numbers verified" if res["verified"] else f"⚠️ unverified: {res['unverified_numbers']}")
                       + f" · {res['latency_s']} s")
            with st.expander("Tools called"):
                st.json(res["tool_calls"])
        st.session_state.history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["reply"]}]

for i in (1, 2, 3, 4, 6):
    with tabs[i]:
        st.info("TODO (Divya) - see README section 6.3")
