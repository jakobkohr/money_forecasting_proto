from __future__ import annotations

from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd


DISCRETIONARY_CATEGORIES = {
    "Weekend activities",
    "Shopping",
    "Eating out",
    "Transport",
}

CATEGORY_LIMITS = {
    "Weekend activities": 0.08,
    "Eating out": 0.10,
    "Shopping": 0.08,
    "Transport": 0.03,
}


def _prepare_tx(tx: pd.DataFrame) -> pd.DataFrame:
    out = tx.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["tags"] = out["tags"].fillna("")
    out["category"] = out["category"].fillna("Unknown")
    out["merchant"] = out["merchant"].fillna("Unknown")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def infer_monthly_income(tx: pd.DataFrame) -> float:
    df = _prepare_tx(tx)
    salary_mask = (
        (df["amount"] > 0)
        & (
            (df["category"].str.lower() == "income")
            | (df["tags"].str.contains("salary", case=False, regex=True))
        )
    )
    salary = df[salary_mask].copy()

    if salary.empty:
        positive = df[df["amount"] > 0].copy()
        if positive.empty:
            return 0.0
        positive["month"] = positive["date"].dt.to_period("M")
        return float(positive.groupby("month")["amount"].sum().median())

    salary["month"] = salary["date"].dt.to_period("M")
    monthly_salary = salary.groupby("month")["amount"].sum()
    return float(monthly_salary.median())


def daily_spend_series(tx: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_tx(tx)
    by_day = (
        df.groupby(df["date"].dt.normalize())
        .agg(
            spend_total=("amount", lambda s: float((-s[s < 0]).sum())),
            income_total=("amount", lambda s: float(s[s > 0].sum())),
        )
        .reset_index()
        .rename(columns={"date": "day"})
    )
    by_day["day"] = pd.to_datetime(by_day["day"])
    return by_day.sort_values("day").reset_index(drop=True)


def compute_category_spend(tx: pd.DataFrame, as_of: pd.Timestamp, days: int = 30) -> pd.DataFrame:
    df = _prepare_tx(tx)
    as_of = pd.to_datetime(as_of)
    window_start = as_of - pd.Timedelta(days=days)
    w = df[(df["date"] > window_start) & (df["date"] <= as_of) & (df["amount"] < 0)].copy()
    if w.empty:
        return pd.DataFrame(columns=["category", "spend"])

    out = (
        w.assign(spend=-w["amount"])
        .groupby("category", as_index=False)["spend"]
        .sum()
        .sort_values("spend", ascending=False)
        .reset_index(drop=True)
    )
    return out


def flag_bad_purchases(tx: pd.DataFrame, monthly_income: float, as_of: pd.Timestamp) -> pd.DataFrame:
    df = _prepare_tx(tx)
    as_of = pd.to_datetime(as_of)
    window_start = as_of - pd.Timedelta(days=30)
    w = df[(df["date"] > window_start) & (df["date"] <= as_of) & (df["amount"] < 0)].copy()
    if w.empty:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "reason", "severity"])

    w["abs_amount"] = -w["amount"]

    category_spend = w.groupby("category")["abs_amount"].sum().to_dict()
    category_median = w.groupby("category")["abs_amount"].median().to_dict()

    flags: list[dict] = []

    for _, row in w.iterrows():
        category = row["category"]
        amt = float(row["abs_amount"])
        tags = str(row["tags"]).lower()
        reasons = []
        severity_score = 0

        if monthly_income > 0 and category in DISCRETIONARY_CATEGORIES and amt > 0.04 * monthly_income:
            reasons.append(f"Single purchase is >4% of monthly income ({amt:.0f}€).")
            severity_score += 3

        limit = CATEGORY_LIMITS.get(category)
        if monthly_income > 0 and limit is not None:
            cap = limit * monthly_income
            actual = float(category_spend.get(category, 0.0))
            if actual > cap:
                reasons.append(f"{category} spend in last 30d ({actual:.0f}€) exceeds recommended {int(limit*100)}% of income.")
                severity_score += 2

        median_cat = float(category_median.get(category, 0.0))
        risky_tag = any(tag in tags for tag in ["overspend", "bottle service", "impulse", "nightlife"])
        if risky_tag and median_cat > 0 and amt > 1.5 * median_cat:
            reasons.append("Tagged as overspend/impulse/nightlife and >1.5x your category median.")
            severity_score += 3

        if reasons:
            severity = "low"
            if severity_score >= 5:
                severity = "high"
            elif severity_score >= 3:
                severity = "medium"

            flags.append(
                {
                    "date": row["date"].date().isoformat(),
                    "merchant": row["merchant"],
                    "category": category,
                    "amount": -amt,
                    "reason": " ".join(reasons),
                    "severity": severity,
                }
            )

    flagged = pd.DataFrame(flags)
    if flagged.empty:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "reason", "severity"])

    severity_rank = {"high": 2, "medium": 1, "low": 0}
    flagged["_sev"] = flagged["severity"].map(severity_rank).fillna(0)
    flagged["_abs"] = flagged["amount"].abs()
    flagged = flagged.sort_values(["_sev", "_abs"], ascending=[False, False]).drop(columns=["_sev", "_abs"])
    return flagged.reset_index(drop=True)


def build_commitments(tx: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_tx(tx)
    recurring = df[(df["is_recurring"] == 1) & (df["amount"] < 0)].copy()
    if recurring.empty:
        return pd.DataFrame(columns=["date", "name", "amount", "source", "tag"])

    def _next_month_date(last_date: pd.Timestamp) -> pd.Timestamp:
        first_next = (last_date + pd.offsets.MonthBegin(1)).normalize()
        day = int(last_date.day)
        max_day = int((first_next + pd.offsets.MonthEnd(0)).day)
        return first_next + pd.Timedelta(days=min(day, max_day) - 1)

    rows = []
    grouped = recurring.groupby(["merchant", "category"], as_index=False)
    for _, g in grouped:
        last_date = g["date"].max()
        amount = float((-g["amount"]).median())
        if amount <= 0:
            continue
        rows.append(
            {
                "date": _next_month_date(last_date),
                "name": f"{g['merchant'].iloc[0]} ({g['category'].iloc[0]})",
                "amount": amount,
                "source": "derived",
                "tag": str(g["category"].iloc[0]).lower(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "name", "amount", "source", "tag"])
    return out.sort_values("date").reset_index(drop=True)


def get_demo_calendar_events() -> pd.DataFrame:
    demo = pd.DataFrame(
        [
            {"date": "2026-03-21", "name": "Birthday gift (Alex)", "amount": 100.0, "source": "calendar", "tag": "birthday"},
            {"date": "2026-03-24", "name": "Credit card repayment", "amount": 620.0, "source": "calendar", "tag": "card_repayment"},
        ]
    )
    demo["date"] = pd.to_datetime(demo["date"])
    return demo


def project_balance(
    tx: pd.DataFrame,
    as_of: pd.Timestamp,
    predicted_daily_spend: Dict[pd.Timestamp, float],
    events: pd.DataFrame,
) -> dict:
    df = _prepare_tx(tx)
    as_of = pd.to_datetime(as_of).normalize()

    hist = df[df["date"] <= as_of]
    if hist.empty:
        current_balance = 0.0
    else:
        current_balance = float(hist["balance_after"].iloc[-1])

    month_end = (as_of + pd.offsets.MonthEnd(0)).normalize()

    events_df = events.copy()
    if events_df.empty:
        events_df = pd.DataFrame(columns=["date", "name", "amount", "source", "tag"])
    if not events_df.empty:
        events_df["date"] = pd.to_datetime(events_df["date"]).dt.normalize()
        events_df["amount"] = pd.to_numeric(events_df["amount"], errors="coerce").fillna(0.0)

    horizon = month_end
    if not events_df.empty:
        horizon = max(month_end, events_df["date"].max())

    date_range = pd.date_range(as_of + timedelta(days=1), horizon, freq="D")
    event_sums = {}
    if not events_df.empty:
        event_sums = events_df.groupby("date")["amount"].sum().to_dict()

    # Project likely monthly income (e.g., salary) so horizons beyond month-end
    # do not become unrealistically negative from spend-only subtraction.
    income_sums = {}
    salary_mask = (
        (df["amount"] > 0)
        & (
            (df["is_recurring"] == 1)
            | (df["category"].str.lower() == "income")
            | (df["tags"].str.contains("salary", case=False, regex=True))
        )
    )
    salary_tx = df[salary_mask].copy()
    if not salary_tx.empty:
        salary_day = int(salary_tx["date"].dt.day.median())
        salary_tx["month"] = salary_tx["date"].dt.to_period("M")
        salary_amount = float(salary_tx.groupby("month")["amount"].sum().median())
        if salary_amount > 0:
            months = pd.period_range(
                start=(as_of + timedelta(days=1)).to_period("M"),
                end=horizon.to_period("M"),
                freq="M",
            )
            for m in months:
                month_start = m.to_timestamp(how="start").normalize()
                max_day = int((month_start + pd.offsets.MonthEnd(0)).day)
                payday = month_start + pd.Timedelta(days=min(salary_day, max_day) - 1)
                if payday > as_of and payday <= horizon:
                    income_sums[payday.normalize()] = income_sums.get(payday.normalize(), 0.0) + salary_amount

    balance = current_balance
    balance_series = []

    for d in date_range:
        balance += float(income_sums.get(d.normalize(), 0.0))
        spend = float(predicted_daily_spend.get(d.normalize(), predicted_daily_spend.get(d, 0.0)))
        balance -= max(0.0, spend)
        balance -= float(event_sums.get(d.normalize(), 0.0))
        balance_series.append({"date": d.normalize(), "projected_balance": balance})

    balance_df = pd.DataFrame(balance_series)

    event_projection = []
    if not events_df.empty:
        for _, row in events_df.sort_values("date").iterrows():
            d = row["date"].normalize()
            bal = current_balance
            if not balance_df.empty:
                hit = balance_df[balance_df["date"] == d]
                if not hit.empty:
                    bal = float(hit.iloc[-1]["projected_balance"])
                elif d > balance_df["date"].max():
                    bal = float(balance_df.iloc[-1]["projected_balance"])
            event_projection.append(
                {
                    "date": d,
                    "name": row.get("name", "Event"),
                    "amount": float(row.get("amount", 0.0)),
                    "projected_balance": bal,
                }
            )

    end_balance = current_balance
    if not balance_df.empty:
        month_hit = balance_df[balance_df["date"] == month_end]
        if not month_hit.empty:
            end_balance = float(month_hit.iloc[-1]["projected_balance"])

    risk_state = "OK"
    if end_balance < 0:
        risk_state = "Will Go Negative"
    elif end_balance < 200:
        risk_state = "At Risk"

    next_obligation_risk = None
    if event_projection:
        upcoming_event = sorted(event_projection, key=lambda x: x["date"])[0]
        if upcoming_event["projected_balance"] < 0:
            next_obligation_risk = f"{upcoming_event['name']} not covered"
        elif upcoming_event["projected_balance"] < upcoming_event["amount"] * 0.5:
            next_obligation_risk = f"Thin buffer for {upcoming_event['name']}"
        else:
            next_obligation_risk = f"{upcoming_event['name']} appears covered"

    return {
        "current_balance": current_balance,
        "month_end": month_end,
        "projected_end_balance": end_balance,
        "risk_state": risk_state,
        "event_projection": event_projection,
        "balance_path": balance_df,
        "next_obligation_risk": next_obligation_risk,
    }
