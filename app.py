from __future__ import annotations

import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import (
    CATEGORY_LIMITS,
    build_commitments,
    compute_category_spend,
    daily_spend_series,
    flag_bad_purchases,
    get_demo_calendar_events,
    infer_monthly_income,
    project_balance,
)
from src.features import add_calendar_features, add_rolling_features, transactions_to_daily
from src.suggestions import generate_shortfall_suggestions, guardrails_from_flags

st.set_page_config(page_title="Money Assistant", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"]  {
        font-family: "Manrope", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1000px 500px at 10% -10%, rgba(56, 189, 248, 0.18), transparent 50%),
            radial-gradient(800px 500px at 90% 0%, rgba(129, 140, 248, 0.20), transparent 45%),
            linear-gradient(160deg, #070b1a 0%, #0b1022 55%, #070b1a 100%);
        color: #e5ebff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b2e 0%, #1b2036 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    [data-testid="stSidebar"] * {
        color: #e6ecff;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        border-radius: 12px;
        padding: 6px 8px;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background: rgba(148, 163, 184, 0.14);
    }
    [data-testid="stDateInput"] input, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
        background: rgba(15, 23, 42, 0.65);
        color: #e5ebff;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 10px;
    }
    .hero-wrap {
        margin-top: 6px;
        margin-bottom: 10px;
        padding: 14px 16px 13px 16px;
        border-radius: 16px;
        background:
            radial-gradient(650px 180px at 0% 0%, rgba(56, 189, 248, 0.20), transparent 60%),
            linear-gradient(160deg, rgba(17, 24, 39, 0.75) 0%, rgba(15, 23, 42, 0.88) 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        margin: 0;
        font-size: clamp(1.02rem, 1.35vw, 1.42rem);
        line-height: 1.3;
        color: #d7e3ff;
        font-weight: 600;
    }
    .mobile-shell {
        max-width: 860px;
        margin: 0 auto;
    }
    .balance-hero {
        margin-top: 8px;
        margin-bottom: 12px;
        text-align: center;
        padding: 16px 10px 10px 10px;
    }
    .balance-value {
        font-size: clamp(2.1rem, 5vw, 4.2rem);
        color: #f7fbff;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .balance-label {
        margin-top: 6px;
        color: #b9c8ea;
        font-size: 1.18rem;
    }
    .balance-projection {
        margin-top: 8px;
        color: #d5e3ff;
        font-size: 1rem;
        font-weight: 600;
    }
    .event-chip {
        margin-top: 10px;
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(248, 113, 113, 0.42);
        border-radius: 14px;
        padding: 12px 14px;
        color: #ffe4e6;
        font-size: 1rem;
    }
    .warning-banner {
        margin-top: 4px;
        margin-bottom: 14px;
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid rgba(252, 165, 165, 0.70);
        box-shadow: 0 14px 34px rgba(153, 27, 27, 0.25);
        background: linear-gradient(120deg, rgba(127, 29, 29, 0.84), rgba(185, 28, 28, 0.64));
        color: #fff1f2;
    }
    .warning-title {
        margin: 0;
        font-size: clamp(1.3rem, 2.1vw, 2rem);
        line-height: 1.12;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    .warning-sub {
        margin-top: 8px;
        font-size: 1.02rem;
        color: #fee2e2;
    }
    .warning-metrics {
        margin-top: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .warning-pill {
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.92rem;
        font-weight: 700;
        color: #fff7ed;
        border: 1px solid rgba(254, 202, 202, 0.55);
        background: rgba(127, 29, 29, 0.44);
    }
    .warning-banner.warning-amber {
        border-color: rgba(251, 191, 36, 0.66);
        box-shadow: 0 14px 34px rgba(120, 53, 15, 0.25);
        background: linear-gradient(120deg, rgba(120, 53, 15, 0.82), rgba(161, 98, 7, 0.58));
    }
    .warning-banner.warning-amber .warning-sub {
        color: #fef3c7;
    }
    .warning-banner.warning-blue {
        border-color: rgba(147, 197, 253, 0.62);
        box-shadow: 0 14px 34px rgba(30, 58, 138, 0.22);
        background: linear-gradient(120deg, rgba(30, 58, 138, 0.76), rgba(30, 64, 175, 0.52));
    }
    .warning-banner.warning-blue .warning-sub {
        color: #dbeafe;
    }
    .action-card {
        margin-top: 12px;
        margin-bottom: 14px;
        border-radius: 16px;
        padding: 14px 16px;
        border: 1px solid rgba(74, 222, 128, 0.44);
        background: linear-gradient(130deg, rgba(6, 78, 59, 0.50), rgba(15, 23, 42, 0.72));
        color: #dcfce7;
    }
    .action-item {
        margin-top: 8px;
        padding: 8px 10px;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(134, 239, 172, 0.28);
        font-size: 0.97rem;
    }
    .tile {
        background: linear-gradient(145deg, rgba(18, 26, 44, 0.90), rgba(12, 18, 34, 0.84));
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 16px;
        padding: 14px 15px;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.34);
        min-height: 106px;
        backdrop-filter: blur(6px);
    }
    .tile-label { color: #b6c3e6; font-size: 0.85rem; margin-bottom: 8px; }
    .tile-value { color: #f1f6ff; font-size: 1.95rem; font-weight: 800; }
    .insight-glow {
        margin-top: 14px;
        margin-bottom: 14px;
        padding: 18px 20px;
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(186, 230, 253, 0.92), rgba(196, 181, 253, 0.88));
        border: 1px solid rgba(191, 219, 254, 0.65);
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .row-card {
        margin-top: 10px;
        border-radius: 14px;
        padding: 12px 14px;
        background: rgba(17, 24, 39, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.20);
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #e6eeff;
    }
    .row-label { color: #c9d5f5; font-size: 0.97rem; }
    .row-value { color: #f8fbff; font-size: 1.22rem; font-weight: 700; }
    .breakdown-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .breakdown-item {
        border-radius: 16px;
        padding: 12px 14px;
        background: rgba(17, 24, 39, 0.66);
        border: 1px solid rgba(148, 163, 184, 0.22);
    }
    .b-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .b-icon {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(99, 102, 241, 0.24);
        color: #dbe6ff;
        font-size: 1rem;
    }
    .b-label {
        color: #d7e2ff;
        font-size: 1.04rem;
    }
    .b-value {
        color: #f8fbff;
        font-size: 1.55rem;
        font-weight: 700;
    }
    .b-trend-up { color: #fda4af; font-weight: 700; }
    .b-trend-flat { color: #cbd5e1; font-weight: 700; }
    .risk-card {
        margin-top: 12px;
        border-radius: 16px;
        padding: 13px 14px;
        border: 1px solid rgba(251, 191, 36, 0.45);
        background: linear-gradient(140deg, rgba(120, 53, 15, 0.35), rgba(31, 41, 55, 0.56));
        color: #fde68a;
        font-size: 1.02rem;
        font-weight: 600;
    }
    .topbar {
        margin-top: 2px;
        margin-bottom: 10px;
        color: #f8fbff;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .subtitle {
        color: #aebcdf;
        margin-top: -4px;
        margin-bottom: 14px;
        font-size: 1rem;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }
    .status-ok { color: #0f766e; font-weight: 600; }
    .status-risk { color: #b45309; font-weight: 600; }
    .status-bad { color: #b91c1c; font-weight: 600; }
    .section-title {
        margin-top: 16px;
        margin-bottom: 8px;
        color: #e8eeff;
        font-size: 1.28rem;
        font-weight: 700;
    }
    @media (max-width: 980px) {
        .tile-value { font-size: 1.45rem; }
        .breakdown-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv("data/synthetic_revolut_transactions.csv")


@st.cache_resource
def load_model_and_meta():
    model = joblib.load("models/daily_spend_model.joblib")
    with open("models/model_meta.json", "r") as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def preprocess_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    out = tx.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["is_recurring"] = pd.to_numeric(out["is_recurring"], errors="coerce").fillna(0).astype(int)
    out["balance_after"] = pd.to_numeric(out["balance_after"], errors="coerce")
    out["merchant"] = out["merchant"].fillna("Unknown")
    out["category"] = out["category"].fillna("Unknown")
    out["tags"] = out["tags"].fillna("")
    out = out.sort_values("date").reset_index(drop=True)
    return out


@st.cache_data
def cached_daily_spend_series(tx: pd.DataFrame) -> pd.DataFrame:
    return daily_spend_series(tx)


@st.cache_data
def cached_category_spend(tx: pd.DataFrame, as_of_dt: pd.Timestamp, days: int = 30) -> pd.DataFrame:
    return compute_category_spend(tx, as_of=as_of_dt, days=days)


@st.cache_data
def cached_flagged_purchases(tx: pd.DataFrame, monthly_income: float, as_of_dt: pd.Timestamp) -> pd.DataFrame:
    return flag_bad_purchases(tx, monthly_income=monthly_income, as_of=as_of_dt)


@st.cache_data
def habit_projection_full_history(tx: pd.DataFrame, as_of_dt: pd.Timestamp) -> dict:
    dfx = tx.copy()
    dfx = dfx[dfx["date"] <= as_of_dt].copy()
    if dfx.empty:
        return {
            "projected_month_spend": 0.0,
            "spend_so_far": 0.0,
            "future_habit_spend": 0.0,
        }

    daily = daily_spend_series(dfx)
    daily["weekday"] = daily["day"].dt.weekday
    weekday_avg = daily.groupby("weekday")["spend_total"].mean().to_dict()
    fallback_avg = float(daily["spend_total"].mean()) if not daily.empty else 0.0

    month_start = as_of_dt.replace(day=1).normalize()
    month_end = (as_of_dt + pd.offsets.MonthEnd(0)).normalize()

    current_month_spend = float(
        -dfx[(dfx["date"] >= month_start) & (dfx["date"] <= as_of_dt) & (dfx["amount"] < 0)]["amount"].sum()
    )

    future_days = pd.date_range(as_of_dt + pd.Timedelta(days=1), month_end, freq="D")
    future_habit_spend = 0.0
    for d in future_days:
        future_habit_spend += float(weekday_avg.get(d.weekday(), fallback_avg))

    projected_month_spend = current_month_spend + future_habit_spend
    return {
        "projected_month_spend": float(projected_month_spend),
        "spend_so_far": float(current_month_spend),
        "future_habit_spend": float(future_habit_spend),
    }


def _tile(label: str, value: str):
    st.markdown(
        f"<div class='tile'><div class='tile-label'>{label}</div><div class='tile-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _row_card(label: str, value: str):
    st.markdown(
        f"<div class='row-card'><span class='row-label'>{label}</span><span class='row-value'>{value}</span></div>",
        unsafe_allow_html=True,
    )


def _breakdown_item(label: str, value: str, icon: str, trend: str = "flat"):
    trend_html = "<span class='b-trend-flat'>-</span>" if trend == "flat" else "<span class='b-trend-up'>↑</span>"
    st.markdown(
        (
            "<div class='breakdown-item'>"
            "<div class='b-top'>"
            f"<span class='b-icon'>{icon}</span>{trend_html}"
            "</div>"
            f"<div class='b-label'>{label}</div>"
            f"<div class='b-value'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _predict_daily_spend_map(
    tx: pd.DataFrame,
    as_of_dt: pd.Timestamp,
    horizon_end: pd.Timestamp,
    model,
    feature_cols: list[str],
    payday_day: int,
) -> dict[pd.Timestamp, float]:
    daily = transactions_to_daily(tx)

    start_day = pd.to_datetime(daily["day"].min())
    end_day = as_of_dt.normalize()
    full_days = pd.DataFrame({"day": pd.date_range(start_day, end_day, freq="D")})

    daily = full_days.merge(daily, on="day", how="left").fillna(
        {"spend_total": 0.0, "income_total": 0.0, "net": 0.0, "balance": 0.0}
    )
    daily = add_calendar_features(daily, payday_day=payday_day)
    daily = add_rolling_features(daily)
    daily = daily.dropna().reset_index(drop=True)

    if horizon_end <= as_of_dt:
        return {}

    future_days = pd.date_range(as_of_dt + pd.Timedelta(days=1), horizon_end, freq="D")
    sim = daily.copy()
    predicted: dict[pd.Timestamp, float] = {}

    for d in future_days:
        next_row = pd.DataFrame(
            [{"day": d, "spend_total": np.nan, "income_total": 0.0, "net": np.nan, "balance": np.nan}]
        )
        sim2 = pd.concat([sim[["day", "spend_total", "income_total", "net", "balance"]], next_row], ignore_index=True)
        sim2 = add_calendar_features(sim2, payday_day=payday_day)
        sim2 = add_rolling_features(sim2)

        x = sim2.iloc[-1:][feature_cols]
        pred = float(model.predict(x)[0])
        pred = max(0.0, pred)

        predicted[d.normalize()] = pred

        sim.loc[len(sim), "day"] = d
        sim.loc[len(sim) - 1, "spend_total"] = pred
        sim.loc[len(sim) - 1, "income_total"] = 0.0
        sim.loc[len(sim) - 1, "net"] = -pred
        sim.loc[len(sim) - 1, "balance"] = np.nan

    return predicted


def _risk_class(risk: str) -> str:
    if risk == "OK":
        return "status-ok"
    if risk == "At Risk":
        return "status-risk"
    return "status-bad"


st.markdown("<div class='topbar'>Monthly Money Forecast</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Banking co-pilot for habit risk and upcoming obligations</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Insights", "Transactions", "Upcoming"], index=0)

st.sidebar.header("Data")
source = st.sidebar.radio("Data source", ["Demo (Revolut sandbox)", "Upload CSV (optional)"], index=0)

if source == "Demo (Revolut sandbox)":
    st.sidebar.success("Calendar (sandbox) connected")
    tx_raw = load_demo_data()
else:
    uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or switch back to Demo.")
        st.stop()
    tx_raw = pd.read_csv(uploaded)

required_cols = {"date", "merchant", "category", "amount", "is_recurring", "balance_after", "tags"}
missing = required_cols - set(tx_raw.columns)
if missing:
    st.error(f"Missing columns: {sorted(missing)}")
    st.stop()

tx = preprocess_transactions(tx_raw)

model, meta = load_model_and_meta()
feature_cols = meta["feature_cols"]
payday_day = int(meta.get("payday_day", 1))

latest_date = tx["date"].max().normalize()
mid_month_candidate = (latest_date.replace(day=1) + pd.Timedelta(days=14)).normalize()
default_as_of = min(mid_month_candidate, latest_date).date()

as_of = st.sidebar.date_input(
    "As of date",
    value=default_as_of,
    min_value=tx["date"].min().date(),
    max_value=tx["date"].max().date(),
)
as_of_dt = pd.to_datetime(as_of)

hist = tx[tx["date"] <= as_of_dt]
if hist.empty:
    st.error("No transactions available up to selected date.")
    st.stop()

monthly_income = infer_monthly_income(tx)
flagged = cached_flagged_purchases(tx, monthly_income=monthly_income, as_of_dt=as_of_dt)
category_30 = cached_category_spend(tx, as_of_dt=as_of_dt, days=30)

default_calendar_events = get_demo_calendar_events().copy()
default_signature = tuple(
    sorted(
        (
            pd.to_datetime(r["date"]).date().isoformat(),
            str(r["name"]),
            float(r["amount"]),
        )
        for _, r in default_calendar_events.iterrows()
    )
)

if "calendar_events" not in st.session_state:
    st.session_state.calendar_events = default_calendar_events.copy()
    st.session_state._calendar_defaults_signature = default_signature
else:
    prev_signature = st.session_state.get("_calendar_defaults_signature")
    if prev_signature != default_signature:
        existing_events = st.session_state.calendar_events.copy()
        if existing_events.empty:
            manual_events = pd.DataFrame(columns=default_calendar_events.columns)
        else:
            existing_events["tag"] = existing_events.get("tag", "").astype(str)
            manual_events = existing_events[existing_events["tag"].str.lower() == "manual"].copy()
        merged_events = pd.concat([default_calendar_events, manual_events], ignore_index=True)
        if not merged_events.empty:
            merged_events["date"] = pd.to_datetime(merged_events["date"]).dt.normalize()
            merged_events = merged_events.drop_duplicates(subset=["date", "name"], keep="first").reset_index(drop=True)
        st.session_state.calendar_events = merged_events
        st.session_state._calendar_defaults_signature = default_signature

derived_commitments = build_commitments(tx)
all_events = pd.concat([st.session_state.calendar_events, derived_commitments], ignore_index=True)
if not all_events.empty:
    all_events["date"] = pd.to_datetime(all_events["date"])
all_events = all_events[all_events["date"] > as_of_dt].sort_values("date").reset_index(drop=True)

month_end = (as_of_dt + pd.offsets.MonthEnd(0)).normalize()
horizon_end = month_end
if not all_events.empty:
    horizon_end = max(month_end, all_events["date"].max())

forecast_key = f"{as_of_dt.date()}|{tx['date'].max().date()}|{len(tx)}|{float(tx['amount'].sum()):.2f}|{horizon_end.date()}"
if st.session_state.get("_forecast_key") != forecast_key:
    predicted_map = _predict_daily_spend_map(
        tx=tx,
        as_of_dt=as_of_dt,
        horizon_end=horizon_end,
        model=model,
        feature_cols=feature_cols,
        payday_day=payday_day,
    )
    st.session_state._forecast_key = forecast_key
    st.session_state._predicted_map = predicted_map

predicted_map = st.session_state.get("_predicted_map", {})
projection = project_balance(tx=tx, as_of=as_of_dt, predicted_daily_spend=predicted_map, events=all_events)
full_history_habits = habit_projection_full_history(tx=tx, as_of_dt=as_of_dt)

days_left_in_month = max(int((month_end - as_of_dt.normalize()).days), 0)
habit_spend_forecast_model = float(
    sum(v for d, v in predicted_map.items() if pd.to_datetime(d).normalize() <= month_end)
)
habit_spend_forecast = habit_spend_forecast_model
if days_left_in_month > 0 and habit_spend_forecast_model < 1.0:
    fallback_history = float(full_history_habits.get("future_habit_spend", 0.0))
    recent_daily = cached_daily_spend_series(tx)
    recent_daily = recent_daily[recent_daily["day"] <= as_of_dt].tail(30).copy()
    baseline_daily = 0.0
    if not recent_daily.empty:
        non_zero_days = recent_daily.loc[recent_daily["spend_total"] > 0, "spend_total"]
        if not non_zero_days.empty:
            baseline_daily = float(non_zero_days.median())
    fallback_recent = baseline_daily * days_left_in_month
    habit_spend_forecast = max(fallback_history, fallback_recent)

upcoming_commitments = float(
    all_events.loc[all_events["date"] <= month_end, "amount"].sum() if not all_events.empty else 0.0
)

next_event = None
if projection["event_projection"]:
    next_event = sorted(projection["event_projection"], key=lambda x: x["date"])[0]

next_calendar_event = None
if projection["event_projection"] and not all_events.empty:
    calendar_events = all_events[all_events["source"] != "derived"][["date", "name", "amount"]].copy()
    if not calendar_events.empty:
        ev_proj = pd.DataFrame(projection["event_projection"])
        calendar_events = calendar_events.merge(ev_proj[["date", "name", "projected_balance"]], on=["date", "name"], how="left")
        if not calendar_events.empty:
            next_calendar_event = calendar_events.sort_values("date").iloc[0].to_dict()

guardrails = guardrails_from_flags(
    flagged=flagged,
    category_spend=category_30,
    monthly_income=monthly_income,
    next_event=next_event,
    as_of=as_of_dt,
)

risk_cls = _risk_class(projection["risk_state"])

if page == "Home":
    st.markdown("<div class='mobile-shell'>", unsafe_allow_html=True)

    focus_event = next_calendar_event or next_event
    days_to_focus_event = None
    projected_at_focus_event = None
    focus_event_amount = None

    if focus_event is not None:
        days_to_focus_event = int((pd.to_datetime(focus_event["date"]).normalize() - as_of_dt.normalize()).days)
        projected_at_focus_event = float(focus_event.get("projected_balance", projection["current_balance"]))
        focus_event_amount = float(focus_event.get("amount", 0.0))

    action_items: list[str] = []
    cat_map = {r["category"]: float(r["spend"]) for _, r in category_30.iterrows()} if not category_30.empty else {}
    weeks_left = max((days_to_focus_event or 28) / 7.0, 1.0)
    for category, limit in CATEGORY_LIMITS.items():
        cat_spend = float(cat_map.get(category, 0.0))
        cap = monthly_income * limit if monthly_income > 0 else 0.0
        if cap > 0 and cat_spend > cap:
            over = cat_spend - cap
            weekly_cut = over / weeks_left
            action_items.append(
                f"{category}: you are €{over:,.0f} above your 30-day guideline. Reduce by about €{weekly_cut:,.0f}/week until the next bill."
            )

    savings_target = max(0.0, -float(projected_at_focus_event)) if projected_at_focus_event is not None else 0.0
    for idea in generate_shortfall_suggestions(tx, as_of=str(as_of_dt.date()), target_savings=savings_target):
        if idea not in action_items:
            action_items.append(idea)
    action_items = action_items[:4]

    st.markdown(
        (
            "<div class='hero-wrap'>"
            f"<h2 class='hero-title'>Welcome back Jakob. Based on your habits, you are likely to spend about "
            f"€{habit_spend_forecast:,.0f} over the next {days_left_in_month} day(s) "
            f"(from {(as_of_dt + pd.Timedelta(days=1)).date()} to {month_end.date()}).</h2>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<div class='balance-hero'>"
            f"<div class='balance-value'>€{projection['current_balance']:,.2f}</div>"
            "<div class='balance-label'>Current Balance</div>"
            f"<div class='balance-projection'>End-of-month projection: €{projection['projected_end_balance']:,.2f}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if focus_event is not None:
        projected_at_event = projected_at_focus_event
        event_amount = focus_event_amount
        if projected_at_event < 0:
            warning_title = f"High-risk alert: {focus_event['name']} may not be covered"
            warning_class = ""
        elif projected_at_event < event_amount * 0.5:
            warning_title = f"Warning: low buffer before {focus_event['name']}"
            warning_class = "warning-amber"
        else:
            warning_title = f"Upcoming event: {focus_event['name']}"
            warning_class = "warning-blue"

        st.markdown(
            (
                f"<div class='warning-banner {warning_class}'>"
                f"<h3 class='warning-title'>{warning_title}</h3>"
                f"<div class='warning-sub'>"
                f"In {max(days_to_focus_event, 0)} day(s), "
                f"{focus_event['name']} is expected to charge €{event_amount:,.0f}. "
                f"Projected balance at that point: €{projected_at_event:,.2f}."
                "</div>"
                "<div class='warning-metrics'>"
                f"<span class='warning-pill'>Event date: {pd.to_datetime(focus_event['date']).date()}</span>"
                f"<span class='warning-pill'>Amount: €{event_amount:,.0f}</span>"
                f"<span class='warning-pill'>Projected balance then: €{projected_at_event:,.2f}</span>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    if action_items:
        action_html = "".join([f"<div class='action-item'>{item}</div>" for item in action_items])
        st.markdown("<div class='section-title'>Action plan to afford upcoming bills</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='action-card'>{action_html}</div>", unsafe_allow_html=True)

    unusual_events = all_events[all_events["source"] != "derived"].copy() if not all_events.empty else pd.DataFrame()

    if unusual_events.empty:
        st.markdown("<div class='section-title'>Upcoming unusual spendings</div>", unsafe_allow_html=True)
        st.write("No unusual spendings detected.")
    else:
        st.markdown("<div class='section-title'>Upcoming unusual spendings</div>", unsafe_allow_html=True)
        ev_proj = pd.DataFrame(projection["event_projection"])
        unusual_view = unusual_events.merge(ev_proj[["date", "name", "projected_balance"]], on=["date", "name"], how="left")
        unusual_view = unusual_view.sort_values("date")
        for _, r in unusual_view.iterrows():
            st.markdown(
                (
                    "<div class='event-chip'>"
                    f"{pd.to_datetime(r['date']).date()} · <strong>{r['name']}</strong> · €{float(r['amount']):,.0f}"
                    f"<br/>Projected balance then: €{float(r['projected_balance']):,.2f}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    cols = st.columns(5)
    with cols[0]:
        _tile("Current balance", f"€{projection['current_balance']:,.2f}")
    with cols[1]:
        _tile("Habit spend forecast (rest of month)", f"€{habit_spend_forecast:,.2f}")
    with cols[2]:
        _tile("Upcoming commitments", f"€{upcoming_commitments:,.2f}")
    with cols[3]:
        _tile("Projected end-of-month balance", f"€{projection['projected_end_balance']:,.2f}")
    with cols[4]:
        _tile("Next obligation risk", projection.get("next_obligation_risk") or "No upcoming obligations")

    st.markdown("<div class='section-title'>Predicted spending breakdown</div>", unsafe_allow_html=True)
    c30_map = {r["category"]: float(r["spend"]) for _, r in category_30.iterrows()} if not category_30.empty else {}
    st.markdown("<div class='breakdown-grid'>", unsafe_allow_html=True)
    _breakdown_item("Eating out", f"€{c30_map.get('Eating out', 0.0):,.0f}", "EO", "up")
    _breakdown_item("Groceries", f"€{c30_map.get('Groceries', 0.0):,.0f}", "GR", "flat")
    _breakdown_item("Transport", f"€{c30_map.get('Transport', 0.0):,.0f}", "TR", "flat")
    _breakdown_item("Weekend activities", f"€{c30_map.get('Weekend activities', 0.0):,.0f}", "WA", "up")
    if next_event is not None:
        _breakdown_item(next_event["name"], f"€{float(next_event['amount']):,.0f}", "EV", "flat")
    st.markdown("</div>", unsafe_allow_html=True)

    if next_event is not None:
        st.markdown(
            (
                "<div class='risk-card'>"
                "Based on your usual spending habits across your full history, "
                f"you’ll likely spend €{habit_spend_forecast:,.0f} for the rest of this month. "
                f"If you keep going like this you will not be able to pay for {next_event['name']} coming up soon."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(f"Risk state: <span class='{risk_cls}'>{projection['risk_state']}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Bad purchases (income-class mismatch)")
    if flagged.empty:
        st.info("No bad purchases flagged in the last 30 days.")
    else:
        st.dataframe(flagged.head(20), use_container_width=True)

    st.subheader("Habit guardrails")
    for g in guardrails:
        st.write(f"- {g}")

elif page == "Insights":
    st.subheader("Spending behavior")

    daily = cached_daily_spend_series(tx)
    daily = daily[daily["day"] <= as_of_dt].copy()
    daily["weekday"] = daily["day"].dt.day_name()
    daily["weekday_idx"] = daily["day"].dt.weekday

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    weekday_avg = (
        daily.groupby(["weekday", "weekday_idx"], as_index=False)["spend_total"].mean().sort_values("weekday_idx")
    )

    weekend_vs = daily.copy()
    weekend_vs["bucket"] = np.where(weekend_vs["day"].dt.weekday >= 5, "Weekend", "Weekday")
    weekend_avg = weekend_vs.groupby("bucket", as_index=False)["spend_total"].mean()

    cat30 = cached_category_spend(tx, as_of_dt=as_of_dt, days=30)

    c1, c2 = st.columns(2)
    with c1:
        chart1 = (
            alt.Chart(weekday_avg)
            .mark_bar(color="#1d4ed8")
            .encode(
                x=alt.X("weekday:N", sort=weekday_order, title="Day"),
                y=alt.Y("spend_total:Q", title="Avg spend (€)"),
                tooltip=["weekday", alt.Tooltip("spend_total:Q", format=".2f")],
            )
            .properties(title="Average Spend by Weekday", height=320)
        )
        st.altair_chart(chart1, use_container_width=True)

    with c2:
        chart2 = (
            alt.Chart(weekend_avg)
            .mark_bar()
            .encode(
                x=alt.X("bucket:N", title=""),
                y=alt.Y("spend_total:Q", title="Avg spend (€)"),
                color=alt.Color("bucket:N", scale=alt.Scale(range=["#64748b", "#ea580c"]), legend=None),
                tooltip=["bucket", alt.Tooltip("spend_total:Q", format=".2f")],
            )
            .properties(title="Weekend vs Weekday Average", height=320)
        )
        st.altair_chart(chart2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        line = (
            alt.Chart(daily)
            .mark_line(color="#0f766e")
            .encode(
                x=alt.X("day:T", title="Date"),
                y=alt.Y("spend_total:Q", title="Daily spend (€)"),
                tooltip=[alt.Tooltip("day:T"), alt.Tooltip("spend_total:Q", format=".2f")],
            )
            .properties(title="Daily Spend Over Time", height=320)
        )
        st.altair_chart(line, use_container_width=True)

    with c4:
        chart4 = (
            alt.Chart(cat30)
            .mark_bar(color="#7c3aed")
            .encode(
                x=alt.X("spend:Q", title="Spend (€)"),
                y=alt.Y("category:N", sort="-x", title="Category"),
                tooltip=["category", alt.Tooltip("spend:Q", format=".2f")],
            )
            .properties(title="Category Breakdown (Last 30 Days)", height=320)
        )
        st.altair_chart(chart4, use_container_width=True)

elif page == "Transactions":
    st.subheader("Transaction feed")

    feed_days = st.slider("Lookback days", min_value=30, max_value=90, value=60, step=10)
    feed = tx[(tx["date"] > as_of_dt - pd.Timedelta(days=feed_days)) & (tx["date"] <= as_of_dt)].copy()

    categories = sorted(feed["category"].unique().tolist()) if not feed.empty else []
    selected_categories = st.multiselect("Category filter", options=categories, default=categories)

    tag_filter = st.radio("Tag filter", ["All", "Weekend only", "Nightlife only", "Late-night only"], horizontal=True)

    if selected_categories:
        feed = feed[feed["category"].isin(selected_categories)]

    if tag_filter == "Weekend only":
        feed = feed[feed["tags"].str.contains("weekend", case=False)]
    elif tag_filter == "Nightlife only":
        feed = feed[feed["tags"].str.contains("nightlife", case=False)]
    elif tag_filter == "Late-night only":
        feed = feed[feed["tags"].str.contains("late_night", case=False)]

    feed["key"] = (
        feed["date"].dt.date.astype(str)
        + "|"
        + feed["merchant"].astype(str)
        + "|"
        + feed["amount"].round(2).astype(str)
    )

    flagged_keys = set()
    if not flagged.empty:
        flagged_copy = flagged.copy()
        flagged_copy["key"] = (
            pd.to_datetime(flagged_copy["date"]).dt.date.astype(str)
            + "|"
            + flagged_copy["merchant"].astype(str)
            + "|"
            + flagged_copy["amount"].round(2).astype(str)
        )
        flagged_keys = set(flagged_copy["key"].tolist())

    feed["flagged"] = feed["key"].isin(flagged_keys)

    view_cols = ["date", "merchant", "category", "amount", "balance_after", "tags", "flagged"]
    feed_view = feed.sort_values("date", ascending=False)[view_cols].reset_index(drop=True)
    st.dataframe(feed_view, use_container_width=True)

elif page == "Upcoming":
    st.subheader("Upcoming obligations")
    st.success("Calendar (sandbox) connected")

    with st.form("add_event_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Event name", value="")
        with c2:
            date_val = st.date_input("Event date", value=as_of_dt.date() + pd.Timedelta(days=10))
        with c3:
            amount = st.number_input("Amount (€)", min_value=1.0, step=10.0, value=100.0)
        submitted = st.form_submit_button("Add event (calendar sandbox)")

    if submitted and name.strip():
        new_row = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime(date_val),
                    "name": name.strip(),
                    "amount": float(amount),
                    "source": "calendar",
                    "tag": "manual",
                }
            ]
        )
        st.session_state.calendar_events = pd.concat([st.session_state.calendar_events, new_row], ignore_index=True)
        st.success("Event added to sandbox calendar.")
        st.rerun()

    upcoming_events = all_events.copy()
    if upcoming_events.empty:
        st.info("No upcoming obligations after selected date.")
    else:
        event_proj = pd.DataFrame(projection["event_projection"])
        event_view = upcoming_events.merge(event_proj[["date", "name", "projected_balance"]], on=["date", "name"], how="left")
        event_view = event_view.sort_values("date")
        st.dataframe(event_view[["date", "name", "amount", "source", "projected_balance"]], use_container_width=True)

    st.subheader("Recommended guardrails")
    for g in guardrails:
        st.write(f"- {g}")

    balance_path = projection.get("balance_path", pd.DataFrame())
    if not balance_path.empty:
        bp = balance_path.copy()
        line = (
            alt.Chart(bp)
            .mark_line(color="#dc2626")
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("projected_balance:Q", title="Projected balance (€)"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("projected_balance:Q", format=".2f")],
            )
            .properties(title="Projected balance path", height=300)
        )
        st.altair_chart(line, use_container_width=True)
