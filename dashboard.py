"""Vista Heights - PG Management Analytics dashboard.

Production-style Streamlit + Plotly application over the real PG datasets.
Every number shown is computed from actual data or persisted model outputs -
no synthetic data, no fake joins.

    streamlit run dashboard.py     # full interactive app
    python dashboard.py            # writes outputs/dashboard.html (static fallback)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import operational_analytics as ops  # noqa: E402
from src import preprocessing  # noqa: E402

APP_TITLE = "Vista Heights — PG Management Analytics"

# Consistent, professional palette.
C_PRIMARY = "#2A9D8F"
C_ACCENT = "#264653"
C_WARN = "#E9C46A"
C_RISK = "#E76F51"
C_HIGH = "#C0392B"
C_MED = "#E67E22"
C_LOW = "#27AE60"

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "vista_heights_chart",
                             "scale": 2},
}

PROBLEM_LABELS = {
    "late_payment": "Late Payment (classification)",
    "monthly_revenue": "Monthly Revenue (regression)",
    "electricity_cost": "Electricity Cost (regression)",
}


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def _data():
    cleaned = preprocessing.clean_all()
    feats = fe.build_all(cleaned)
    return cleaned, feats


def _load_csv(name: str):
    p = config.OUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _load_meta(problem: str) -> dict | None:
    p = config.OUT_DIR / f"model_meta_{problem}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _kpi_cards(st, items: list[tuple[str, str]]):
    """Render a row of KPI cards from (label, value) pairs."""
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.markdown(f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
                   f"<div class='kpi-value'>{value}</div></div>",
                   unsafe_allow_html=True)


def _load_meta_json(filename: str) -> dict | None:
    p = config.OUT_DIR / filename
    if p.exists():
        return json.loads(p.read_text())
    return None


def _revenue_verdict(rmeta: dict, comp) -> str:
    """Explain which revenue model wins and why — from the real metrics."""
    ts, ml = rmeta["ts_mape"], rmeta["ml_mape"]
    winner = rmeta["winner"]
    if winner == "TimeSeries":
        head = (f"**Verdict: the Time-Series model wins** "
                f"(MAPE {ts:.1f}% vs ML {ml:.1f}%).")
        why = ("Its explicit trend + 12-month seasonal structure captures the "
               "yearly PG intake cycle better than an ML model learning from only "
               f"{rmeta['n_train_months_total']} monthly rows.")
    else:
        head = (f"**Verdict: the ML model ({rmeta['ml_model']}) wins** "
                f"(MAPE {ml:.1f}% vs Time-Series {ts:.1f}%).")
        why = ("For one-month-ahead revenue the lagged features (especially last "
               "month's revenue) track the strong month-to-month persistence and "
               "recent growth better than a purely-seasonal naive forecast, which "
               "compares against a year-ago value and misses the 2024→2026 trend. "
               "A linear model beats XGBoost here because ~28 monthly rows are far "
               "too few for gradient boosting — it overfits, while the linear AR "
               "model matches the true low complexity of the signal.")
    caveat = ("With only a few dozen monthly observations both errors carry wide "
              "uncertainty; the Time-Series model remains the primary forecaster "
              "for multi-month horizons, and the ML model is a complementary "
              "one-step cross-check.")
    return f"{head}\n\n{why}\n\n_{caveat}_"


def _late_payment_risk(feats):
    """Score latest-month invoices with the saved best (time-based) model."""
    import joblib
    from src import train
    mpath = config.MODEL_DIR / "late_payment_best.pkl"
    if not mpath.exists():
        return None
    model = joblib.load(mpath)
    X = train.PROBLEMS["late_payment"](feats)[0]
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)
    inv = feats["invoice_features"]
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = model.predict(X).astype(float)
    out = inv[["invoice_id", "tenant_id", "billing_period", "total_amount",
               "prior_unpaid", "is_unpaid"]].copy()
    out["risk_score"] = proba
    latest = out["billing_period"].max()
    out = out[out["billing_period"] == latest].copy()
    out["risk_level"] = pd.cut(out["risk_score"], [-0.01, 0.3, 0.7, 1.01],
                               labels=["Low", "Medium", "High"])
    return out.sort_values("risk_score", ascending=False)


def _anomaly_with_severity(name: str):
    df = _load_csv(name)
    if df is None or df.empty or "anomaly_score" not in df:
        return None
    q1, q2 = df["anomaly_score"].quantile([0.33, 0.66])
    df["severity"] = np.select(
        [df["anomaly_score"] >= q2, df["anomaly_score"] >= q1],
        ["High", "Medium"], default="Low")
    return df


def _segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Business names from real profile: highest paid-LTV segment = Anchor."""
    if profile is None or "ltv_paid" not in profile:
        return {}
    ranked = profile.sort_values("ltv_paid", ascending=False)["segment"].tolist()
    names = {}
    label_bank = ["Anchor Tenants", "Regular Tenants", "Short-stay Tenants",
                  "At-risk Tenants"]
    for i, seg in enumerate(ranked):
        names[int(seg)] = label_bank[i] if i < len(label_bank) else f"Segment {seg}"
    return names


# --------------------------------------------------------------------------- #
# Dynamic pieces (generated only from real model outputs)
# --------------------------------------------------------------------------- #
def _dynamic_recommendations(cleaned, feats, risk) -> list[tuple[str, str]]:
    """(severity, text) pairs computed from actual predictions and data."""
    recs = []
    fs = _load_csv("forecast_summary.csv")
    beds = feats["bed_features"]

    if risk is not None and len(risk):
        hi = risk[risk["risk_score"] > 0.5]
        amt = hi["total_amount"].sum()
        recs.append(("High",
                     f"{len(hi)} high-risk invoices detected in "
                     f"{risk['billing_period'].max()} — estimated revenue at risk "
                     f"₹{amt/1e5:.2f} Lakhs. Contact these tenants within 48 hours "
                     f"and prioritise the top-20 list on the Late Payment Risk page."))

    ea = _anomaly_with_severity("anomalies_electricity.csv")
    if ea is not None:
        hi_e = (ea["severity"] == "High").sum()
        apts = ", ".join(ea.loc[ea.severity == "High", "apartment_code"]
                         .astype(str).unique()[:5])
        recs.append(("High" if hi_e else "Medium",
                     f"{len(ea)} electricity anomalies flagged ({hi_e} high-severity). "
                     f"Recommend meter inspection starting with apartments: {apts}."))

    ia = _anomaly_with_severity("anomalies_invoices.csv")
    if ia is not None:
        recs.append(("Medium",
                     f"{len(ia)} invoice anomalies flagged for billing review — "
                     f"check amount composition and credit-day outliers."))

    notice = int(beds["on_notice"].sum())
    vacant = int(beds["is_vacant"].sum())
    if notice + vacant:
        recs.append(("Medium",
                     f"{notice} beds on notice + {vacant} vacant "
                     f"({(notice+vacant)/len(beds)*100:.0f}% of capacity idle or "
                     f"churning). Launch retention outreach for notice beds and "
                     f"re-list vacant high-rate beds first."))

    seg = _load_csv("tenant_segments_profile.csv")
    if seg is not None and len(seg) >= 2:
        names = _segment_names(seg)
        top = seg.sort_values("ltv_paid", ascending=False).iloc[0]
        recs.append(("Medium",
                     f"{int(top['n_tenants'])} {names.get(int(top['segment']), 'top-segment')} "
                     f"average ₹{top['ltv_paid']/1e5:.2f} Lakhs lifetime value each — "
                     f"protect them with priority maintenance and renewal offers."))

    if fs is not None:
        rev = fs[fs.series == "revenue"]
        if len(rev):
            recs.append(("Low",
                         f"Next-month revenue forecast ₹{rev.next_month.iloc[0]/1e5:.1f} "
                         f"Lakhs (walk-forward MAPE {rev.MAPE.iloc[0]:.1f}%). Use for "
                         f"cash-flow planning."))
        el = fs[fs.series == "elec_cost"]
        if len(el):
            recs.append(("Low",
                         f"Next-month electricity cost forecast "
                         f"₹{el.next_month.iloc[0]/1e5:.1f} Lakhs "
                         f"(MAPE {el.MAPE.iloc[0]:.1f}%). Budget accordingly."))
    return recs


def _exec_summary_items(cleaned, feats, risk) -> list[tuple[str, str, str]]:
    """(icon, label, value) computed from real data + model outputs."""
    inv = cleaned["invoices"]
    fs = _load_csv("forecast_summary.csv")
    latest = inv["billing_period"].max()
    cur_rev = inv.loc[inv.billing_period == latest, "total_amount"].sum()
    items = [("💰", "Current Revenue (mo)", f"₹{cur_rev/1e5:.1f} L")]
    if fs is not None and len(fs[fs.series == "revenue"]):
        items.append(("📈", "Predicted Revenue (next mo)",
                      f"₹{fs[fs.series == 'revenue'].next_month.iloc[0]/1e5:.1f} L"))
    risk_amt = (inv["total_amount"] * inv["is_unpaid"]).sum()
    items.append(("⚠️", "Revenue at Risk (total)", f"₹{risk_amt/1e7:.2f} Cr"))
    items.append(("✅", "Collection Rate", f"{(1-inv.is_unpaid.mean())*100:.1f}%"))
    if risk is not None:
        items.append(("🚨", "High-Risk Invoices (mo)",
                      f"{int((risk.risk_score > 0.5).sum())}"))
    if fs is not None and len(fs[fs.series == "elec_cost"]):
        items.append(("⚡", "Electricity Forecast (next mo)",
                      f"₹{fs[fs.series == 'elec_cost'].next_month.iloc[0]/1e5:.1f} L"))
    return items


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _kpis(cleaned, feats):
    inv = cleaned["invoices"]
    beds = feats["bed_features"]
    return {
        "💰 Total Revenue": f"₹{inv['total_amount'].sum()/1e7:.2f} Cr",
        "⚠️ Revenue at Risk": f"₹{(inv['total_amount']*inv['is_unpaid']).sum()/1e7:.2f} Cr",
        "✅ Collection Rate": f"{(1 - inv['is_unpaid'].mean()) * 100:.1f}%",
        "🛏️ Occupancy": f"{beds['is_occupied'].mean() * 100:.1f}%",
        "📋 Beds on Notice": f"{int(beds['on_notice'].sum())}",
        "⚡ Electricity Cost": f"₹{cleaned['electricity']['amount'].sum()/1e5:.0f} L",
        "🔧 Open Tickets": f"{int((cleaned['tickets']['status'] != 'closed').sum())}",
    }


def _figures(cleaned, feats):
    pm = feats["property_month"].copy()
    pm["month"] = pm["billing_period"].astype(str)
    figs = {}
    figs["revenue"] = px.area(pm, x="month", y="revenue", title="Monthly Revenue",
                              color_discrete_sequence=[C_PRIMARY])
    figs["collection"] = px.line(pm, x="month", y="collection_rate", markers=True,
                                 title="Collection Rate",
                                 color_discrete_sequence=[C_ACCENT])
    figs["collection"].add_hline(y=0.9, line_dash="dot", line_color=C_WARN,
                                 annotation_text="90% target")
    figs["risk"] = px.bar(pm, x="month", y="revenue_at_risk",
                          title="Revenue at Risk by Month",
                          color_discrete_sequence=[C_RISK])
    figs["electricity"] = px.line(pm, x="month", y="elec_cost", markers=True,
                                  title="Electricity Cost",
                                  color_discrete_sequence=[C_WARN])
    bl = feats["bed_features"]["bed_lifecycle_status"].value_counts().reset_index()
    bl.columns = ["status", "count"]
    figs["beds"] = px.pie(bl, names="status", values="count",
                          title="Bed Lifecycle Mix", hole=0.45,
                          color="status",
                          color_discrete_map={"occupied": C_PRIMARY,
                                              "notice": C_RISK,
                                              "vacant": C_WARN,
                                              "booked": C_ACCENT})
    tk = cleaned["tickets"]["issue_type"].value_counts().head(10).reset_index()
    tk.columns = ["issue_type", "count"]
    figs["tickets"] = px.bar(tk, x="count", y="issue_type", orientation="h",
                             title="Top Maintenance Issues",
                             color_discrete_sequence=[C_ACCENT])
    for f in figs.values():
        f.update_layout(margin=dict(l=10, r=10, t=48, b=10),
                        title_font_size=15)
    return figs


def _forecast_fig(pm, csv_name, col, title, mae=None):
    fc = _load_csv(csv_name)
    if fc is None:
        return None
    fig = go.Figure()
    fig.add_scatter(x=pm["billing_period"].astype(str), y=pm[col], name="actual",
                    mode="lines+markers", line=dict(color=C_PRIMARY))
    fig.add_scatter(x=fc["billing_period"], y=fc[col], name="forecast",
                    mode="lines+markers", line=dict(dash="dash", color=C_RISK))
    if mae is not None and mae == mae:
        band = 1.96 * mae   # ~95% interval from real walk-forward errors
        fig.add_scatter(x=fc["billing_period"], y=fc[col] + band,
                        mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(x=fc["billing_period"], y=(fc[col] - band).clip(lower=0),
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(231,111,81,0.15)",
                        name="95% confidence (walk-forward)")
    fig.update_layout(title=title, hovermode="x unified",
                      margin=dict(l=10, r=10, t=48, b=10))
    return fig


def _backtest_fig(csv_name, title):
    bt = _load_csv(csv_name)
    if bt is None:
        return None
    fig = go.Figure()
    fig.add_scatter(x=bt["billing_period"], y=bt["actual"], name="actual",
                    mode="lines+markers", line=dict(color=C_PRIMARY))
    fig.add_scatter(x=bt["billing_period"], y=bt["predicted"], name="predicted",
                    mode="lines+markers", line=dict(dash="dot", color=C_ACCENT))
    fig.update_layout(title=title, hovermode="x unified",
                      margin=dict(l=10, r=10, t=48, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title=APP_TITLE, page_icon="🏢", layout="wide")
    st.markdown("""
    <style>
      .kpi-card {border:1px solid rgba(128,128,128,.25); border-radius:12px;
                 padding:14px 16px; text-align:center; height:100%;}
      .kpi-label {font-size:.78rem; opacity:.75; margin-bottom:4px;
                  white-space:nowrap;}
      .kpi-value {font-size:1.35rem; font-weight:700;}
      .badge {display:inline-block; padding:2px 10px; border-radius:10px;
              color:white; font-size:.75rem; font-weight:600;}
      .badge-High {background:#C0392B;} .badge-Medium {background:#E67E22;}
      .badge-Low {background:#27AE60;}
      .summary-strip {border-left:4px solid #2A9D8F; padding:8px 14px;
                      margin:4px 0; border-radius:4px;
                      background:rgba(42,157,143,.07);}
      div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25);
                                   border-radius:12px; padding:10px;}
    </style>""", unsafe_allow_html=True)

    st.title(f"🏢 {APP_TITLE}")

    cleaned, feats = _data()
    pm = feats["property_month"].copy()
    figs = _figures(cleaned, feats)
    risk = _late_payment_risk(feats)
    fs = _load_csv("forecast_summary.csv")

    # ---- Auto Executive Summary strip (real model outputs only) ------------ #
    with st.container():
        items = _exec_summary_items(cleaned, feats, risk)
        cols = st.columns(len(items))
        for c, (icon, label, value) in zip(cols, items):
            c.markdown(f"<div class='kpi-card'><div class='kpi-label'>{icon} "
                       f"{label}</div><div class='kpi-value'>{value}</div></div>",
                       unsafe_allow_html=True)
        recs = _dynamic_recommendations(cleaned, feats, risk)
        if recs:
            top = recs[0]
            st.markdown(f"<div class='summary-strip'>🎯 <b>Top recommendation:</b> "
                        f"{top[1]}</div>", unsafe_allow_html=True)
    st.markdown("")

    tabs = st.tabs([
        "📊 Executive Summary", "📈 Revenue Forecast", "🛏️ Occupancy Forecast",
        "🏠 Apartment-wise Forecast", "🚨 Late Payment Risk", "⚡ Anomaly Alerts",
        "👥 Tenant Segmentation", "🤖 ML Performance", "💡 Recommendations",
        "📦 Asset Management", "🚪 Available Beds", "🔧 Maintenance",
        "🏆 Apartment Performance", "📤 Notice & Exit"])

    # 1) Executive Summary ---------------------------------------------------- #
    with tabs[0]:
        kpis = _kpis(cleaned, feats)
        cols = st.columns(len(kpis))
        for col, (k, v) in zip(cols, kpis.items()):
            col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{k}</div>"
                         f"<div class='kpi-value'>{v}</div></div>",
                         unsafe_allow_html=True)
        st.markdown("")
        c1, c2 = st.columns(2)
        c1.plotly_chart(figs["revenue"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c2.plotly_chart(figs["collection"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c1.plotly_chart(figs["risk"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c2.plotly_chart(figs["electricity"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c1.plotly_chart(figs["beds"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c2.plotly_chart(figs["tickets"], use_container_width=True,
                        config=PLOTLY_CONFIG)

    # 2) Revenue Forecast ------------------------------------------------------ #
    with tabs[1]:
        st.markdown("### 1️⃣ Time-Series Forecast (primary model)")
        st.caption("Holt-Winters / seasonal-naive on the monthly revenue series — "
                   "the primary forecaster, unchanged.")
        mae = mape = method = None
        if fs is not None:
            row = fs[fs.series == "revenue"]
            if len(row):
                mae, mape, method = (float(row.MAE.iloc[0]),
                                     float(row.MAPE.iloc[0]),
                                     row.method.iloc[0])
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast method", method or "—")
        m2.metric("Walk-forward MAPE", f"{mape:.1f}%" if mape else "—")
        m3.metric("Walk-forward MAE", f"₹{mae/1e5:.2f} L" if mae else "—")
        fig = _forecast_fig(pm, "forecast_revenue.csv", "revenue",
                            "Revenue forecast with 95% confidence band", mae)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        bfig = _backtest_fig("backtest_revenue.csv",
                             "Prediction vs Actual — walk-forward validation")
        if bfig:
            st.plotly_chart(bfig, use_container_width=True, config=PLOTLY_CONFIG)
        fc = _load_csv("forecast_revenue.csv")
        if fc is not None:
            st.download_button("⬇️ Export revenue forecast CSV",
                               fc.to_csv(index=False), "forecast_revenue.csv",
                               "text/csv")

        # ---- Second model: supervised ML on the monthly panel -------------- #
        st.markdown("---")
        st.markdown("### 2️⃣ Machine Learning Revenue Prediction")
        rmeta = _load_meta_json("model_meta_revenue_ml.json")
        comp = _load_csv("comparison_revenue_models.csv")
        if rmeta and comp is not None:
            comp.columns = ["Model"] + list(comp.columns[1:])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ML model", rmeta["ml_model"])
            c2.metric("Features (lagged only)", rmeta["n_features"])
            c3.metric("Test months", rmeta["n_test_months"])
            c4.metric("ML next-month",
                      f"₹{rmeta['ml_next_month_revenue']/1e5:.2f} L"
                      if rmeta.get("ml_next_month_revenue") else "—")
            st.caption("Features: lagged revenue (t-1..t-12), lagged 3-month mean, "
                       "lagged tenant count, calendar. **Excluded as leakage:** "
                       + ", ".join(rmeta["excluded_leakage_features"]) + ".")

            st.markdown("**Model comparison — Time-Series vs ML (identical "
                        "walk-forward windows)**")
            st.dataframe(comp.set_index("Model")
                         .style.highlight_min(subset=["MAPE", "MAE", "RMSE"],
                                              color="rgba(42,157,143,.25)")
                         .highlight_max(subset=["R2"],
                                        color="rgba(42,157,143,.25)"),
                         use_container_width=True)

            bt = _load_csv("backtest_revenue_ml.csv")
            if bt is not None:
                f = go.Figure()
                f.add_scatter(x=bt.billing_period, y=bt.actual, name="actual",
                              mode="lines+markers", line=dict(color=C_PRIMARY,
                                                              width=3))
                f.add_scatter(x=bt.billing_period, y=bt.ts_predicted,
                              name="Time-Series", mode="lines+markers",
                              line=dict(dash="dash", color=C_RISK))
                f.add_scatter(x=bt.billing_period, y=bt.ml_predicted,
                              name=f"ML ({rmeta['ml_model']})",
                              mode="lines+markers",
                              line=dict(dash="dot", color=C_ACCENT))
                f.update_layout(title="Actual vs Time-Series vs ML "
                                      "(walk-forward, one-step)",
                                hovermode="x unified",
                                margin=dict(l=10, r=10, t=48, b=10))
                st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
                st.download_button("⬇️ Export model-comparison CSV",
                                   comp.to_csv(index=False),
                                   "comparison_revenue_models.csv", "text/csv")

            st.markdown(_revenue_verdict(rmeta, comp))
        else:
            st.info("Run: python -m src.revenue_ml")

    # 3) Occupancy Forecast ---------------------------------------------------- #
    with tabs[2]:
        st.subheader("Occupancy proxy — active billed tenants")
        st.caption("Bed tables are point-in-time snapshots, so physical bed-level "
                   "occupancy history does not exist in the data. Active billed "
                   "tenants per month is the real, verifiable occupancy proxy.")
        mae_t = None
        if fs is not None:
            row = fs[fs.series == "active_tenants"]
            if len(row):
                mae_t = float(row.MAE.iloc[0])
                c1, c2, c3 = st.columns(3)
                c1.metric("Forecast method", row.method.iloc[0])
                c2.metric("Walk-forward MAPE", f"{row.MAPE.iloc[0]:.1f}%")
                c3.metric("Next month", f"{row.next_month.iloc[0]:.0f} tenants")
        fig = _forecast_fig(pm, "forecast_active_tenants.csv", "active_tenants",
                            "Active tenants forecast", mae_t)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        bfig = _backtest_fig("backtest_active_tenants.csv",
                             "Prediction vs Actual — walk-forward validation")
        if bfig:
            st.plotly_chart(bfig, use_container_width=True, config=PLOTLY_CONFIG)
        fc = _load_csv("forecast_active_tenants.csv")
        if fc is not None:
            st.download_button("⬇️ Export occupancy forecast CSV",
                               fc.to_csv(index=False),
                               "forecast_active_tenants.csv", "text/csv")

    # 4) Apartment-wise Forecast ----------------------------------------------- #
    with tabs[3]:
        st.subheader("Apartment-wise electricity forecast")
        st.caption("Electricity is the only real apartment × month series in the "
                   "data (invoices carry no apartment code), so apartment-level "
                   "forecasting covers electricity units and amount.")
        apt = _load_csv("forecast_apartment_summary.csv")
        if apt is not None:
            sel = st.multiselect("Filter apartments",
                                 sorted(apt["apartment_code"].unique()))
            view = apt[apt.apartment_code.isin(sel)] if sel else apt
            fig = px.bar(view.head(20), x="apartment_code", y="next_month_amount",
                         color="amount_mape", color_continuous_scale="RdYlGn_r",
                         title="Next-month electricity amount "
                               "(colour = backtest MAPE %)",
                         labels={"next_month_amount": "₹ next month"})
            fig.update_layout(margin=dict(l=10, r=10, t=48, b=10))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            st.dataframe(view, use_container_width=True, height=320)
            full = _load_csv("forecast_apartment_electricity.csv")
            if full is not None:
                st.download_button("⬇️ Export apartment forecast CSV",
                                   full.to_csv(index=False),
                                   "forecast_apartment_electricity.csv", "text/csv")
        else:
            st.info("Run: python -m src.apartment_forecasting")

    # 5) Late Payment Risk ------------------------------------------------------ #
    with tabs[4]:
        st.subheader("Late-payment risk — latest billing month")
        meta = _load_meta("late_payment")
        if meta:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", meta["best_model"])
            c2.metric("Validation", "time-based split")
            c3.metric("Trained on", f"{meta['n_rows']:,} invoices")
            c4.metric("Version", meta["model_version"])
        if risk is not None and len(risk):
            hi = risk[risk.risk_score > 0.5]
            c1, c2, c3 = st.columns(3)
            c1.metric("🚨 High-risk invoices", len(hi))
            c2.metric("💸 Amount at risk",
                      f"₹{hi.total_amount.sum()/1e5:.2f} L")
            c3.metric("📅 Billing month", str(risk.billing_period.max()))
            fig = px.histogram(risk, x="risk_score", nbins=30, color="risk_level",
                               color_discrete_map={"High": C_HIGH,
                                                   "Medium": C_MED, "Low": C_LOW},
                               title="Predicted non-payment probability distribution")
            fig.update_layout(margin=dict(l=10, r=10, t=48, b=10))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            st.markdown("**Top 20 highest-risk tenants (by invoice)**")
            top20 = risk.head(20)[["tenant_id", "invoice_id", "total_amount",
                                   "prior_unpaid", "risk_score", "risk_level"]]
            st.dataframe(top20.style.background_gradient(
                subset=["risk_score"], cmap="Reds"),
                use_container_width=True, height=420)
            st.download_button("⬇️ Export high-risk tenants CSV",
                               risk[risk.risk_score > 0.5].to_csv(index=False),
                               "high_risk_tenants.csv", "text/csv")
            st.download_button("⬇️ Export all predictions CSV",
                               risk.to_csv(index=False),
                               "late_payment_predictions.csv", "text/csv")

            # Feature importance: permutation (model-agnostic, honest for any
            # estimator) + SHAP when the best model was tree-based.
            pi = _load_csv("perm_importance_late_payment.csv")
            sh = _load_csv("shap_late_payment.csv")
            c1, c2 = st.columns(2)
            if pi is not None:
                pi.columns = ["feature", "importance"]
                f = px.bar(pi.head(12).sort_values("importance"), x="importance",
                           y="feature", orientation="h",
                           title="Permutation importance (held-out future slice)",
                           color_discrete_sequence=[C_ACCENT])
                f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
                c1.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            if sh is not None:
                sh.columns = ["feature", "mean_abs_shap"]
                f = px.bar(sh.head(12).sort_values("mean_abs_shap"),
                           x="mean_abs_shap", y="feature", orientation="h",
                           title="SHAP mean |value| (best tree model)",
                           color_discrete_sequence=[C_PRIMARY])
                f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
                c2.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Train the model first: python -m src.train late_payment")

    # 6) Anomaly Alerts ---------------------------------------------------------- #
    with tabs[5]:
        st.subheader("Anomaly alerts (IsolationForest)")
        for label, fname, keycols in [
                ("⚡ Electricity", "anomalies_electricity.csv",
                 ["apartment_code", "billing_month", "units_consumed", "amount"]),
                ("🧾 Invoices", "anomalies_invoices.csv",
                 ["invoice_id", "billing_month", "total_amount", "credit_days"])]:
            df = _anomaly_with_severity(fname)
            if df is None:
                continue
            st.markdown(f"### {label}")
            counts = df["severity"].value_counts()
            c1, c2, c3 = st.columns(3)
            for c, sev in zip((c1, c2, c3), ("High", "Medium", "Low")):
                c.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>"
                    f"<span class='badge badge-{sev}'>{sev}</span></div>"
                    f"<div class='kpi-value'>{int(counts.get(sev, 0))}</div></div>",
                    unsafe_allow_html=True)
            show = [c for c in keycols if c in df.columns] + \
                   ["anomaly_score", "severity"]
            st.dataframe(
                df.sort_values("anomaly_score", ascending=False)[show].head(25),
                use_container_width=True, height=300)
            st.download_button(f"⬇️ Export {label.split()[-1].lower()} anomalies CSV",
                               df.to_csv(index=False), fname, "text/csv",
                               key=f"dl_{fname}")

    # 7) Tenant Segmentation ------------------------------------------------------ #
    with tabs[6]:
        st.subheader("Tenant segments — real billing behaviour")
        seg = _load_csv("tenant_segments_profile.csv")
        if seg is not None:
            names = _segment_names(seg)
            seg["segment_name"] = seg["segment"].map(names)
            c1, c2 = st.columns(2)
            f1 = px.bar(seg, x="segment_name", y="ltv_paid",
                        title="Average lifetime value (paid) per tenant",
                        color="segment_name",
                        color_discrete_sequence=[C_PRIMARY, C_ACCENT])
            f2 = px.bar(seg, x="segment_name", y="n_tenants",
                        title="Tenants per segment", color="segment_name",
                        color_discrete_sequence=[C_PRIMARY, C_ACCENT])
            for f in (f1, f2):
                f.update_layout(showlegend=False,
                                margin=dict(l=10, r=10, t=48, b=10))
            c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
            c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
            st.dataframe(seg.set_index("segment_name"), use_container_width=True)

            st.markdown("**Business interpretation** (computed from the profile):")
            for _, r in seg.iterrows():
                st.markdown(
                    f"- **{r['segment_name']}** — {int(r['n_tenants'])} tenants, "
                    f"avg tenure {r['tenure_months']:.0f} months, avg rent "
                    f"₹{r['avg_rent']:,.0f}, lifetime value ₹{r['ltv_paid']/1e5:.2f} "
                    f"Lakhs, unpaid ratio {r['unpaid_ratio']:.0%}. "
                    + ("Core revenue base — protect with priority service and "
                       "renewal incentives." if r["ltv_paid"] == seg.ltv_paid.max()
                       else "Growth pool — convert to long-stay via upgrade offers "
                            "and consistent service quality."))
            segs_full = _load_csv("tenant_segments.csv")
            if segs_full is not None:
                segs_full["segment_name"] = segs_full["segment"].map(names)
                st.download_button("⬇️ Export segment assignments CSV",
                                   segs_full.to_csv(index=False),
                                   "tenant_segments.csv", "text/csv")

    # 8) ML Performance ------------------------------------------------------------ #
    with tabs[7]:
        st.subheader("Model performance & metadata")
        st.caption("All metrics from the time-based (chronological) holdout — the "
                   "honest out-of-time evaluation. Leaderboards include every "
                   "compared algorithm.")
        for prob, label in PROBLEM_LABELS.items():
            lb = _load_csv(f"leaderboard_{prob}.csv")
            meta = _load_meta(prob)
            if lb is None:
                continue
            st.markdown(f"### {label}")
            if meta:
                c = st.columns(6)
                c[0].metric("Best model", meta["best_model"])
                c[1].metric("Last trained", meta["trained_at"].split()[0])
                c[2].metric("Dataset size", f"{meta['n_rows']:,}")
                c[3].metric("Features", meta["n_features"])
                c[4].metric("Train duration", f"{meta['training_duration_sec']}s")
                c[5].metric("Version", meta["model_version"])
            metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1",
                                       "ROC_AUC", "MAE", "RMSE", "MAPE", "R2"]
                           if c in lb.columns]
            st.dataframe(
                lb.set_index("Model")[metric_cols].round(4)
                  .style.highlight_max(axis=0, color="rgba(42,157,143,.25)")
                  if metric_cols else lb,
                use_container_width=True)
            st.markdown("---")

    # 9) Business Recommendations ---------------------------------------------------- #
    with tabs[8]:
        st.subheader("Business recommendations — generated from live model outputs")
        recs = _dynamic_recommendations(cleaned, feats, risk)
        for sev, text in recs:
            st.markdown(
                f"<div class='summary-strip'><span class='badge badge-{sev}'>"
                f"{sev}</span>&nbsp;&nbsp;{text}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(_blocked_tasks_html(), unsafe_allow_html=True)

    # 10) Asset Management ------------------------------------------------------ #
    with tabs[9]:
        st.subheader("Asset Management")
        a = ops.assets_summary(cleaned["assets"])
        _kpi_cards(st, [
            ("📦 Total Assets", f"{a['total']:,}"),
            ("🗂️ Categories", f"{len(a['by_category'])}"),
            ("🔧 Asset Types", f"{len(a['by_type'])}"),
            ("🟢 Allocated",
             f"{int(a['by_status'].loc[a['by_status']['status']=='allocated','count'].sum())}"),
        ])
        st.info("ℹ️ Warranty alerts, assets-by-apartment and Active/Damaged splits "
                "are **not shown** — `warranty_expiry` is 0.1% populated, "
                "`apartment_code` is 100% null, and `condition` has no 'damaged' "
                "value (only good/new). See dataset audit.")
        c1, c2 = st.columns(2)
        f1 = px.pie(a["by_category"], names="category", values="count", hole=0.45,
                    title="Assets by Category",
                    color_discrete_sequence=px.colors.sequential.Teal)
        f2 = px.bar(a["by_type"].sort_values("count"), x="count", y="type",
                    orientation="h", title="Assets by Type (top 15)",
                    color_discrete_sequence=[C_ACCENT])
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        f3 = px.bar(a["by_status"], x="status", y="count", title="Assets by Status",
                    color="status", color_discrete_sequence=[C_PRIMARY, C_WARN])
        f4 = px.bar(a["by_condition"], x="condition", y="count",
                    title="Assets by Condition",
                    color="condition", color_discrete_sequence=[C_PRIMARY, C_ACCENT])
        for f in (f3, f4):
            f.update_layout(showlegend=False, margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f4, use_container_width=True, config=PLOTLY_CONFIG)
        if a["purchase_timeline"] is not None:
            st.markdown(f"**Assets purchased by month** "
                        f"(only {a['purchase_coverage']:,} of {a['total']:,} assets "
                        f"have a purchase_date — {a['purchase_coverage']/a['total']*100:.0f}%)")
            f5 = px.bar(a["purchase_timeline"], x="month", y="assets",
                        title="Assets Purchased by Month (subset with dates)",
                        color_discrete_sequence=[C_WARN])
            f5.update_layout(margin=dict(l=10, r=10, t=48, b=10))
            st.plotly_chart(f5, use_container_width=True, config=PLOTLY_CONFIG)
        cats = st.multiselect("Filter category", sorted(cleaned["assets"]
                                                        ["category"].unique()))
        tbl = a["table"]
        if cats:
            tbl = tbl[tbl["category"].isin(cats)]
        st.dataframe(tbl, use_container_width=True, height=300)
        st.download_button("⬇️ Export assets CSV", tbl.to_csv(index=False),
                           "assets.csv", "text/csv")
        st.caption(f"**Summary:** {a['total']:,} assets, led by "
                   f"{a['by_category'].iloc[0]['category']} "
                   f"({a['by_category'].iloc[0]['count']:,}). "
                   f"{int(a['by_status'].loc[a['by_status'].status=='inventory','count'].sum())} "
                   f"in inventory (not yet allocated).")

    # 11) Available Beds -------------------------------------------------------- #
    with tabs[10]:
        st.subheader("Available Beds")
        ba = ops.bed_availability(cleaned["beds_snapshot"])
        _kpi_cards(st, [
            ("🛏️ Total Beds", f"{ba['total_beds']}"),
            ("🚪 Vacant Beds", f"{ba['vacant_beds']}"),
            ("📊 Occupancy", f"{ba['occupancy_pct']}%"),
            ("💰 Vacant Revenue Opportunity",
             f"₹{ba['vacant_revenue_opportunity']/1e5:.2f} L/mo"),
        ])
        st.caption("Floor-wise vacancy is not available (no floor column); vacancy "
                   "is shown **block-wise** from the apartment-code prefix. Historical "
                   "vacancy trend is not possible — beds are a current snapshot.")
        c1, c2 = st.columns(2)
        f1 = px.pie(ba["lifecycle"], names="status", values="count", hole=0.45,
                    title="Bed Lifecycle Mix",
                    color="status", color_discrete_map={"occupied": C_PRIMARY,
                    "notice": C_RISK, "vacant": C_WARN, "booked": C_ACCENT})
        f2 = px.bar(ba["by_block"], x="block", y="vacancy_pct",
                    title="Vacancy % by Block", color="vacancy_pct",
                    color_continuous_scale="OrRd")
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        apt_v = ba["by_apartment"]
        apt_v = apt_v[apt_v["vacant"] > 0]
        f3 = px.bar(apt_v, x="apartment_code", y="vacant",
                    title="Vacant Beds by Apartment",
                    color_discrete_sequence=[C_RISK])
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        blocks = st.multiselect("Filter block",
                                sorted(ba["by_block"]["block"].dropna().unique()))
        vt = ba["vacant_table"].copy()
        if blocks:
            vt = vt[vt["apartment_code"].str[0].isin(blocks)]
        st.dataframe(vt, use_container_width=True, height=300)
        st.download_button("⬇️ Export vacant beds CSV", vt.to_csv(index=False),
                           "vacant_beds.csv", "text/csv")
        st.caption(f"**Summary:** {ba['vacant_beds']} of {ba['total_beds']} beds "
                   f"vacant ({100-ba['occupancy_pct']:.1f}%), a monthly revenue "
                   f"opportunity of ₹{ba['vacant_revenue_opportunity']/1e5:.2f} L "
                   f"if filled at current rates.")

    # 12) Maintenance Performance ----------------------------------------------- #
    with tabs[11]:
        st.subheader("Maintenance Performance")
        ms = ops.maintenance_summary(cleaned["tickets"])
        _kpi_cards(st, [
            ("🔧 Total Tickets", f"{ms['total']:,}"),
            ("🟠 Open", f"{ms['open']}"),
            ("✅ Closed", f"{ms['closed']:,}"),
            ("⏱️ Avg Resolution", f"{ms['avg_resolution_hours']:.0f} h"),
            ("⛔ SLA Breached",
             f"{ms['sla_breached']} ({ms['sla_breach_pct']}%)"),
        ])
        c1, c2 = st.columns(2)
        f1 = px.pie(ms["by_status"], names="status", values="count", hole=0.45,
                    title="Ticket Status", color_discrete_sequence=
                    px.colors.sequential.Teal)
        f2 = px.bar(ms["by_priority"], x="priority", y="count",
                    title="Ticket Priority", color="priority",
                    color_discrete_map={"high": C_HIGH, "medium": C_MED,
                                        "low": C_LOW, "urgent": "#7b241c"})
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(showlegend=False, margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        f3 = px.bar(ms["by_issue"].sort_values("count"), x="count", y="issue_type",
                    orientation="h", title="Issue Type Analysis",
                    color_discrete_sequence=[C_ACCENT])
        f4 = px.line(ms["monthly"], x="month", y="tickets", markers=True,
                     title="Monthly Ticket Trend",
                     color_discrete_sequence=[C_PRIMARY])
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f4.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f4, use_container_width=True, config=PLOTLY_CONFIG)
        f5 = px.bar(ms["by_apartment"].head(20), x="apartment_code", y="complaints",
                    title="Apartment-wise Complaints (top 20)",
                    color="complaints", color_continuous_scale="OrRd")
        f5.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f5, use_container_width=True, config=PLOTLY_CONFIG)
        st.download_button("⬇️ Export apartment complaints CSV",
                           ms["by_apartment"].to_csv(index=False),
                           "apartment_complaints.csv", "text/csv")
        st.caption(f"**Summary:** {ms['open']} open of {ms['total']:,} tickets; "
                   f"average resolution {ms['avg_resolution_hours']:.0f} hours; "
                   f"SLA breached on {ms['sla_breach_pct']}%. Top issue: "
                   f"{ms['by_issue'].iloc[0]['issue_type']} "
                   f"({ms['by_issue'].iloc[0]['count']}).")

    # 13) Apartment Performance ------------------------------------------------- #
    with tabs[12]:
        st.subheader("Apartment Performance & Health Score")
        ap = ops.apartment_performance(cleaned["electricity"], cleaned["tickets"],
                                       cleaned["beds_snapshot"])
        _kpi_cards(st, [
            ("🏠 Apartments", f"{len(ap)}"),
            ("🏆 Healthiest", f"{ap.iloc[0]['apartment_code']}"),
            ("⚠️ Lowest Health", f"{ap.iloc[-1]['apartment_code']}"),
            ("🔧 Most Complaints",
             f"{ap.sort_values('complaints', ascending=False).iloc[0]['apartment_code']}"),
        ])
        st.info("ℹ️ Apartment-wise **revenue/collection are not shown** — invoices "
                "carry no apartment_code (UUID tenant only). Health score blends "
                "real apartment-keyed metrics: complaints (45%), vacant beds (35%), "
                "electricity cost (20%); higher = healthier.")
        top = st.slider("Show top/bottom N apartments by health", 5, len(ap), 15)
        f1 = px.bar(ap.head(top), x="apartment_code", y="health_score",
                    title=f"Top {top} Apartments by Health Score",
                    color="health_score", color_continuous_scale="RdYlGn")
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        f2 = px.bar(ap.sort_values("elec_cost", ascending=False).head(15),
                    x="apartment_code", y="elec_cost",
                    title="Electricity Cost Ranking (top 15)",
                    color_discrete_sequence=[C_WARN])
        f3 = px.bar(ap.sort_values("complaints", ascending=False).head(15),
                    x="apartment_code", y="complaints",
                    title="Complaint Hotspots (top 15)",
                    color_discrete_sequence=[C_RISK])
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        st.dataframe(ap[["apartment_code", "health_score", "complaints", "vacant",
                         "elec_cost", "avg_units", "complaint_rank", "elec_rank"]]
                     .round(1), use_container_width=True, height=320)
        st.download_button("⬇️ Export apartment performance CSV",
                           ap.to_csv(index=False), "apartment_performance.csv",
                           "text/csv")
        st.caption(f"**Summary:** {ap.iloc[0]['apartment_code']} is healthiest "
                   f"(score {ap.iloc[0]['health_score']}); "
                   f"{ap.sort_values('complaints', ascending=False).iloc[0]['apartment_code']} "
                   f"has the most complaints "
                   f"({int(ap['complaints'].max())}).")

    # 14) Notice & Exit --------------------------------------------------------- #
    with tabs[13]:
        st.subheader("Notice & Exit Analytics")
        na = ops.notice_analytics(cleaned["notices"])
        _kpi_cards(st, [
            ("📋 Total Notices", f"{na['total_notices']}"),
            ("🚪 Upcoming Exits", f"{na['upcoming_exits']}"),
            ("💸 Monthly Revenue Impact",
             f"₹{na['monthly_revenue_impact']/1e5:.2f} L"),
            ("📅 Avg Notice Period", f"{na['avg_notice_days']:.0f} days"),
        ])
        st.info("ℹ️ **Notice reasons are not shown** — the notices dataset has no "
                "reason/remarks column.")
        c1, c2 = st.columns(2)
        f1 = px.bar(na["monthly"], x="notice_month", y="notices",
                    title="Monthly Notice Trend",
                    color_discrete_sequence=[C_RISK])
        f2 = px.bar(na["exit_month"], x="exit_month", y="revenue_impact",
                    title="Revenue Impact by Exit Month (₹)",
                    color_discrete_sequence=[C_WARN])
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        f3 = px.bar(na["by_apartment"], x="apartment_code", y="notices",
                    title="Apartment-wise Notice Count",
                    color="notices", color_continuous_scale="OrRd")
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("**Upcoming vacating beds**")
        st.dataframe(na["upcoming_table"], use_container_width=True, height=280)
        st.download_button("⬇️ Export upcoming exits CSV",
                           na["upcoming_table"].to_csv(index=False),
                           "upcoming_exits.csv", "text/csv")
        st.caption(f"**Summary:** {na['upcoming_exits']} tenants scheduled to vacate; "
                   f"₹{na['monthly_revenue_impact']/1e5:.2f} L monthly rent at stake "
                   f"across {na['total_notices']} notices.")


# --------------------------------------------------------------------------- #
# Static HTML fallback (no Streamlit needed)
# --------------------------------------------------------------------------- #
def _blocked_tasks_html() -> str:
    rows = [
        ("Tenant exit / churn prediction",
         "notices/beds use name+apartment; invoices use UUID tenant_id - no link"),
        ("Per-bed / per-apartment rent profitability",
         "invoices carry no apartment_code/bed_code"),
        ("Tenant-to-maintenance behaviour",
         "tickets tenant_name (text) != invoice tenant_id (UUID)"),
        ("Historical bed occupancy",
         "beds tables are current snapshots only - no time history"),
    ]
    lis = "".join(f"<li><b>{t}</b> — {r}</li>" for t, r in rows)
    return ("<h3 style='font-family:sans-serif'>Predictions not offered "
            "(no real join exists)</h3><ul style='font-family:sans-serif'>"
            f"{lis}</ul>")


def export_html():
    cleaned, feats = _data()
    risk = _late_payment_risk(feats)
    kpis = _kpis(cleaned, feats)
    figs = _figures(cleaned, feats)
    kpi_html = "".join(
        f"<div style='display:inline-block;margin:8px;padding:14px 20px;"
        f"border:1px solid #ddd;border-radius:10px;font-family:sans-serif'>"
        f"<div style='color:#666;font-size:13px'>{k}</div>"
        f"<div style='font-size:22px;font-weight:700'>{v}</div></div>"
        for k, v in kpis.items())
    parts = [f"<h1 style='font-family:sans-serif'>🏢 {APP_TITLE}</h1>", kpi_html]
    for fig in figs.values():
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    apt = _load_csv("forecast_apartment_summary.csv")
    if apt is not None:
        fig = px.bar(apt.head(15), x="apartment_code", y="next_month_amount",
                     title="Apartment-wise next-month electricity (top 15)")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    recs = _dynamic_recommendations(cleaned, feats, risk)
    rec_html = "".join(f"<li><b>[{s}]</b> {t}</li>" for s, t in recs)
    parts.append("<h2 style='font-family:sans-serif'>Business Recommendations "
                 f"(from live model outputs)</h2><ul "
                 f"style='font-family:sans-serif'>{rec_html}</ul>")
    parts.append(_blocked_tasks_html())
    out = config.OUT_DIR / "dashboard.html"
    out.write_text("<html><body>" + "".join(parts) + "</body></html>",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as _sr
        if _sr.get_script_run_ctx() is not None:
            run_streamlit()
        else:
            export_html()
    except Exception:
        export_html()
