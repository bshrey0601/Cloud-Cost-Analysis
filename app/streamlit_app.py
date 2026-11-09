"""Cloud Cost Analyst -- Streamlit BI app.

    streamlit run app/streamlit_app.py

Tabs: Overview | Forecast | Anomalies | Optimization | Ask the Analyst | Model Validation
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from cca import anomaly as A  # noqa: E402
from cca import forecasting as F  # noqa: E402
from cca import optimization as O  # noqa: E402
from cca.agent import CostAnalystAgent  # noqa: E402
from cca.llm import get_llm  # noqa: E402
from cca.pricing import PRICE_SOURCE  # noqa: E402
from cca.warehouse import active_db_path, connect  # noqa: E402

RESULTS = ROOT / "eval" / "results"

# Validated categorical order (blue, orange, aqua) + reserved status red for anomalies.
SERIES = {"Delay-insensitive": "#2a78d6", "Interactive": "#eb6834", "Unknown": "#1baf7a"}
BLUE, CRITICAL, MUTED, GRID = "#2a78d6", "#d03b3b", "#52514e", "#e6e5e0"

st.set_page_config(page_title="Cloud Cost Analyst", layout="wide")


# ----------------------------------------------------------------- helpers
def money(v: float) -> str:
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e6:
        return f"{s}${a / 1e6:,.2f}M"
    if a >= 1e3:
        return f"{s}${a / 1e3:,.1f}K"
    return f"{s}${a:,.0f}"


def style(fig: go.Figure, height: int = 360, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=48, b=8), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0b0b0b"),
        title=dict(y=0.98, yanchor="top", x=0, xanchor="left", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="top", y=-0.24, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED), title=y_title,
                     title_font=dict(color=MUTED, size=12))
    return fig


@st.cache_data(show_spinner=False)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = connect()
    try:
        return con.execute(sql, list(params)).df()
    finally:
        con.close()


@st.cache_data(show_spinner="Running the forecasting model ...")
def cached_forecast(scope: str, horizon_days: int):
    if scope == "total":
        y = q("SELECT cost_usd FROM mart_hourly_total ORDER BY hour_idx").cost_usd.to_numpy()
    else:
        y = q("SELECT cost_usd FROM mart_hourly_category WHERE vm_category = ? ORDER BY hour_idx",
              (scope,)).cost_usd.to_numpy()
    fc = F.forecast_with_intervals(y, horizon_days * 24, level=0.9)
    return y, fc


@st.cache_data(show_spinner="Scanning the top 50 subscriptions for anomalies ...")
def cached_anomalies():
    df = q("SELECT subscription_alias s, hour_idx, cost_usd FROM mart_hourly_top_subscription")
    piv = df.pivot(index="hour_idx", columns="s", values="cost_usd").reindex(range(720)).fillna(0.0)
    det = A.RobustSeasonalDetector()
    events, scores = [], {}
    for c in piv.columns:
        sc = det.score(piv[c].to_numpy())
        scores[c] = sc
        events += [e.as_dict() for e in A.events_from_flags(sc, c)]
    ev = pd.DataFrame(events).sort_values("excess_cost_usd", key=abs, ascending=False)
    return ev, scores


@st.cache_data(show_spinner=False)
def cached_opt(commit_discount: float = 0.0, spot_discount: float = 0.0) -> dict:
    con = connect()
    try:
        return O.recommendations(con, commit_discount=commit_discount, spot_discount=spot_discount)
    finally:
        con.close()


def load_json(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------- header
# Streamlit Community Cloud: API keys set under App settings -> Secrets become env vars here.
try:
    for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_MODEL", "ANTHROPIC_MODEL",
               "CCA_LLM_PROVIDER"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file configured -- offline mode
    pass


@st.cache_resource(show_spinner="Preparing the data warehouse (first run only) ...")
def _warehouse() -> str:
    return str(active_db_path())


try:
    _warehouse()
except FileNotFoundError as exc:
    st.error(f"Data not found: {exc}")
    st.stop()

st.title("Cloud Cost Analyst")
st.caption("Microsoft Azure Public Dataset V2 — 30 days of Microsoft's own (first-party) Azure VM usage, "
           f"2.7M VMs across 6,687 internal subscriptions · {PRICE_SOURCE}")

tabs = st.tabs(["Overview", "Forecast", "Anomalies", "Optimization", "Ask the Analyst", "Model Validation"])

# ------------------------------------------------------------- overview
with tabs[0]:
    daily = q("SELECT trace_day, vm_category, cost_usd, used_cost_usd FROM mart_daily_category ORDER BY 1")
    total = daily.cost_usd.sum()
    opt = cached_opt()
    ev, _ = cached_anomalies()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("30-day spend (list price)", money(total))
    c2.metric("Average daily spend", money(total / 30))
    c3.metric("Hard savings available", money(opt["hard_savings_usd"]),
              f"{opt['hard_savings_usd'] / total:.1%} of spend", delta_color="off")
    c4.metric("Anomaly events (top 50 subs)", f"{len(ev)}")

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for cat, color in SERIES.items():
            d = daily[daily.vm_category == cat]
            fig.add_trace(go.Scatter(x=d.trace_day, y=d.cost_usd, name=cat, mode="lines",
                                     line=dict(color=color, width=2),
                                     hovertemplate="Day %{x}<br>%{y:$,.0f}<extra>" + cat + "</extra>"))
            last = d.iloc[-1]
            fig.add_annotation(x=last.trace_day, y=last.cost_usd, text=cat, showarrow=False, xanchor="left",
                               xshift=6, font=dict(color=MUTED, size=12))
        fig.update_layout(title="Daily spend by workload category", hovermode="x unified")
        fig.update_xaxes(title="Trace day", range=[1, 36.5])
        st.plotly_chart(style(fig, y_title="USD per day"), width="stretch")
    with right:
        top = q("SELECT subscription_alias, total_spend_usd, primary_category FROM dim_subscription "
                "ORDER BY total_spend_usd DESC LIMIT 10")
        fig = go.Figure(go.Bar(x=top.total_spend_usd, y=top.subscription_alias, orientation="h",
                               marker=dict(color=BLUE, cornerradius=4),
                               text=[money(v) for v in top.total_spend_usd], textposition="outside",
                               textfont=dict(color=MUTED),
                               hovertemplate="%{y}<br>%{x:$,.0f}<extra></extra>"))
        fig.update_layout(title="Top 10 subscriptions by 30-day spend")
        fig.update_xaxes(range=[0, top.total_spend_usd.max() * 1.22])
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(style(fig), width="stretch")

    util = daily.groupby("vm_category")[["cost_usd", "used_cost_usd"]].sum()
    util["avg_cpu_weighted_util"] = util.used_cost_usd / util.cost_usd
    st.markdown("**Spend vs. CPU-weighted utilisation** — the share of each dollar that bought busy CPU time.")
    st.dataframe(data=util.rename(columns={"cost_usd": "Spend (USD)", "used_cost_usd": "CPU-busy spend (USD)",
                                      "avg_cpu_weighted_util": "Utilisation"})
                 .style.format({"Spend (USD)": "${:,.0f}", "CPU-busy spend (USD)": "${:,.0f}", "Utilisation": "{:.1%}"}),
                 width="stretch")

# -------------------------------------------------------------- forecast
with tabs[1]:
    c1, c2 = st.columns([2, 3])
    scope = c1.selectbox("Scope", ["total", "Delay-insensitive", "Interactive", "Unknown"])
    horizon = c2.slider("Horizon (days beyond day 30)", 1, 14, 7)
    y, fc = cached_forecast(scope, horizon)
    hist_h = list(range(720 - 14 * 24, 720))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[h / 24 + 1 for h in hist_h], y=y[hist_h], name="Actual", mode="lines",
                             line=dict(color="#0b0b0b", width=2),
                             hovertemplate="Day %{x:.1f}<br>%{y:$,.0f}/h<extra>Actual</extra>"))
    fx = [h / 24 + 1 for h in fc.hours]
    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=list(fc.upper) + list(fc.lower[::-1]), fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="90% interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=fc.point, name="Forecast", mode="lines", line=dict(color=BLUE, width=2),
                             hovertemplate="Day %{x:.1f}<br>%{y:$,.0f}/h<extra>Forecast</extra>"))
    fig.update_layout(title=f"Hourly spend: last 14 days and {horizon}-day forecast ({scope})", hovermode="x unified")
    fig.update_xaxes(title="Trace day")
    st.plotly_chart(style(fig, 420, "USD per hour"), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Forecast, next {horizon} days", money(fc.total_point))
    c2.metric("90% range", f"{money(fc.total_lower)} – {money(fc.total_upper)}")
    c3.metric("Forecast daily average", money(fc.total_point / horizon),
              f"{(fc.total_point / horizon) / (y[-168:].sum() / 7) - 1:+.1%} vs last 7 days", delta_color="off")
    st.caption(fc.method)

# -------------------------------------------------------------- anomalies
with tabs[2]:
    ev, scores = cached_anomalies()
    st.markdown("Robust seasonal detector (median baseline + MAD scale + materiality floor of ≈$2K/month), "
                "run on the hourly spend of the 50 largest subscriptions.")
    show = ev.copy()
    show["window"] = show.apply(lambda r: f"day {r.start_hour // 24 + 1}, {r.start_hour % 24:02d}:00 · {r.duration_hours} h", axis=1)
    st.dataframe(hide_index=True, data=show[["series", "direction", "pattern", "window", "excess_cost_usd", "pct_vs_expected", "peak_z"]]
                 .rename(columns={"series": "Subscription", "direction": "Type", "pattern": "Pattern", "window": "When",
                                  "excess_cost_usd": "Excess vs expected", "pct_vs_expected": "Deviation",
                                  "peak_z": "Peak robust z"})
                 .style.format({"Excess vs expected": "${:,.0f}", "Deviation": "{:+.1%}", "Peak robust z": "{:,.1f}"}),
                 width="stretch", height=260)
    pick = st.selectbox("Inspect a subscription", list(dict.fromkeys(ev.series.tolist() + sorted(scores))))
    sc = scores[pick]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc.hour_idx / 24 + 1, y=sc.actual, name="Actual", mode="lines",
                             line=dict(color="#0b0b0b", width=2),
                             hovertemplate="Day %{x:.2f}<br>%{y:$,.2f}/h<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(x=sc.hour_idx / 24 + 1, y=sc.expected, name="Expected", mode="lines",
                             line=dict(color=BLUE, width=2, dash="dot"),
                             hovertemplate="Day %{x:.2f}<br>%{y:$,.2f}/h<extra>Expected</extra>"))
    flagged = sc[sc.flag]
    fig.add_trace(go.Scatter(x=flagged.hour_idx / 24 + 1, y=flagged.actual, name="Anomalous hour", mode="markers",
                             marker=dict(color=CRITICAL, size=8, symbol="triangle-up",
                                         line=dict(color="#fcfcfb", width=2)),
                             hovertemplate="Day %{x:.2f}<br>%{y:$,.2f}/h<extra>Anomaly</extra>"))
    fig.update_layout(title=f"{pick}: hourly spend vs. expected", hovermode="x unified")
    fig.update_xaxes(title="Trace day")
    st.plotly_chart(style(fig, 380, "USD per hour"), width="stretch")

# ----------------------------------------------------------- optimization
with tabs[3]:
    st.markdown("Savings over the 30-day window (≈ one month of run-rate). Idle and rightsizing use only list "
                "prices; **commitment and Spot discounts are your inputs** — set them to your contract rates.")
    c1, c2 = st.columns(2)
    cd = c1.slider("Commitment discount (input)", 0.0, 0.7, 0.0, 0.05, format="%.2f")
    sd = c2.slider("Spot discount (input)", 0.0, 0.9, 0.0, 0.05, format="%.2f")
    r = cached_opt(cd, sd)
    summ = r["summary"].copy()
    summ["vms"] = summ["vms"].astype("float")
    st.dataframe(hide_index=True, data=summ.rename(columns={"lever": "Lever", "vms": "VMs", "addressable_cost_usd": "Addressable spend",
                                      "savings_usd": "Savings", "pct_of_total_spend": "Addressable % of spend",
                                      "basis": "Basis"})
                 .style.format({"Addressable spend": "${:,.0f}", "Savings": "${:,.0f}",
                                "Addressable % of spend": "{:.1%}", "VMs": "{:,.0f}"}, na_rep="—"),
                 width="stretch")
    st.info("Levers overlap: Spot candidates include VMs that are also rightsizing candidates, so do not add "
            "Spot savings to rightsizing savings. Hard savings (idle + rightsizing) = "
            f"**{money(r['hard_savings_usd'])}** ({r['hard_savings_usd'] / r['total_cost_usd']:.1%} of spend).")
    bs = r["by_subscription"].head(15)
    fig = go.Figure(go.Bar(x=bs.savings_usd, y=bs.subscription_alias, orientation="h",
                           marker=dict(color=BLUE, cornerradius=4), text=[money(v) for v in bs.savings_usd],
                           textposition="outside", textfont=dict(color=MUTED),
                           hovertemplate="%{y}<br>%{x:$,.0f}<extra></extra>"))
    fig.update_layout(title="Hard savings by subscription (top 15)")
    fig.update_xaxes(range=[0, bs.savings_usd.max() * 1.15])
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(style(fig, 480), width="stretch")

# --------------------------------------------------------- ask the analyst
with tabs[4]:
    llm = get_llm()
    mode = f"{llm.name} ({getattr(llm, 'model', '')})" if llm else "offline (rule planner + template writer)"
    st.markdown(f"**Agent mode:** {mode}. Set `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` to enable the LLM planner "
                "and writer. Every memo is checked by the deterministic Auditor before you see it.")
    examples = ["Why did spend change in days 18-20 compared with days 11-17?",
                "Which subscriptions had unusual cost spikes?",
                "Forecast total spend for the next 7 days",
                "Where can we save money?",
                "What are the top workload categories by spend in week 2?"]
    ex = st.selectbox("Example questions", ["(type your own below)"] + examples)
    question = st.text_input("Question", value="" if ex.startswith("(") else ex)
    if st.button("Analyze", type="primary") and question.strip():
        with st.spinner("Planning, querying, writing and auditing ..."):
            res = CostAnalystAgent(llm=llm).answer(question)
        badge = "✅ Audit PASSED" if res.audit.passed else "❌ Audit FAILED"
        st.markdown(f"### Memo  \n{badge} — {res.audit.summary().splitlines()[0]}  \n"
                    f"Revisions: {res.revisions} · Fell back to template: {res.fell_back} · {res.seconds:.1f}s")
        if res.first_draft_audit and not res.first_draft_audit.passed:
            with st.expander("What the Auditor caught in the first draft"):
                st.text(res.first_draft_audit.summary())
        st.markdown(res.memo)
        with st.expander("Fact ledger (every number above, with its source)"):
            st.dataframe(hide_index=True, data=pd.DataFrame(res.ledger.as_records())[["id", "label", "display", "tool", "args"]],
                         width="stretch")
        with st.expander("Agent trace"):
            st.json(res.steps)

# ------------------------------------------------------- model validation
with tabs[5]:
    st.subheader("Data reconciliation")
    st.dataframe(hide_index=True, data=q("SELECT * FROM dq_reconciliation"), width="stretch")

    st.subheader("Forecasting backtest (rolling origin, days 16-30)")
    fs = RESULTS / "forecast_summary.csv"
    if fs.exists():
        s = pd.read_csv(fs)
        for h, lab in [(24, "Next-day"), (168, "Next-week")]:
            t = s[s.horizon_hours == h].pivot(index="model", columns="series", values="daily_total_mape")
            st.markdown(f"**{lab} forecast — error on the period total (MAPE)**")
            st.dataframe(data=t.style.format("{:.2%}").highlight_min(axis=0, color="#cde2fb"), width="stretch")
    st.subheader("Anomaly detection (held-out subscriptions, injected anomalies)")
    an = load_json("anomaly_eval.json")
    if an:
        st.dataframe(data=pd.DataFrame({k: {"F1": v["f1"], "Precision": v["precision"], "Recall": v["recall"],
                                       "Alarms per series-week on normal data": v["clean_alarms_per_series_week"]}
                                   for k, v in an["detectors"].items()}).T.style.format("{:.3f}"),
                     width="stretch")
    st.subheader("Auditor fault injection")
    fi = load_json("auditor_fault_injection.json")
    if fi:
        st.dataframe(data=pd.DataFrame(fi["by_fault"]).T.style.format({"rate": "{:.1%}"}), width="stretch")
    st.subheader("Agent benchmark vs independent ground truth")
    for f in sorted(RESULTS.glob("benchmark_*.json")):
        st.json(json.loads(f.read_text())["summary"])
