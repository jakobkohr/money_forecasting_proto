import pandas as pd
import numpy as np
from datetime import datetime

def _month_start_end(dt: pd.Timestamp):
    start = dt.replace(day=1)
    # next month start
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1, day=1)
    else:
        next_month = dt.replace(month=dt.month + 1, day=1)
    end = next_month - pd.Timedelta(days=1)
    return start, end

def forecast_remaining_spend(df: pd.DataFrame, as_of: str | None = None) -> dict:
    """
    Simple forecast:
    - compute average daily spend by weekday from last 8 weeks (excluding Income)
    - project remaining days in current month
    - add known recurring commitments remaining in month (subscriptions/rent/etc.)
    Returns expected, low, high + confidence.
    """
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx = dfx.sort_values("date")

    if as_of is None:
        as_of_dt = dfx["date"].max()
    else:
        as_of_dt = pd.to_datetime(as_of)

    m_start, m_end = _month_start_end(as_of_dt)
    days_remaining = pd.date_range(as_of_dt + pd.Timedelta(days=1), m_end, freq="D")

    # last 8 weeks window for behavior
    window_start = as_of_dt - pd.Timedelta(days=56)
    hist = dfx[(dfx["date"] >= window_start) & (dfx["date"] <= as_of_dt)].copy()

    # treat spending as negative amounts excluding Income category / positive amounts
    spend = hist[(hist["amount"] < 0) & (hist["category"].str.lower() != "income")].copy()
    if spend.empty:
        return {"expected": 0, "low": 0, "high": 0, "confidence": "low", "days_remaining": len(days_remaining)}

    spend["weekday"] = spend["date"].dt.weekday
    spend["daily_spend"] = spend["amount"].abs()

    # average daily spend by weekday
    weekday_avg = spend.groupby("weekday")["daily_spend"].mean()

    # variability for low/high band
    weekday_std = spend.groupby("weekday")["daily_spend"].std().fillna(0)

    expected = 0.0
    variance = 0.0
    for d in days_remaining:
        wd = d.weekday()
        mu = float(weekday_avg.get(wd, weekday_avg.mean()))
        sd = float(weekday_std.get(wd, weekday_std.mean()))
        expected += mu
        variance += (sd ** 2)

    # recurring commitments remaining in month (use is_recurring or detected)
    month_future = dfx[(dfx["date"] > as_of_dt) & (dfx["date"] <= m_end)]
    recurring_future = month_future[(month_future.get("is_recurring", 0) == 1) | (month_future.get("recurring_detected", 0) == 1)]
    recurring_spend = recurring_future[recurring_future["amount"] < 0]["amount"].abs().sum()

    expected_total = expected + float(recurring_spend)

    # uncertainty band (simple)
    sd_total = float(np.sqrt(variance))
    low = max(0.0, expected_total - 0.8 * sd_total)
    high = expected_total + 0.8 * sd_total

    # confidence heuristic
    months_of_data = max(1, (dfx["date"].max() - dfx["date"].min()).days // 30)
    recent_days = (as_of_dt - window_start).days
    vol = spend["daily_spend"].std()
    confidence = "high"
    if months_of_data < 3 or recent_days < 28:
        confidence = "medium"
    if vol > 80 or months_of_data < 2:
        confidence = "low"

    return {
        "expected": round(expected_total, 2),
        "low": round(low, 2),
        "high": round(high, 2),
        "confidence": confidence,
        "days_remaining": len(days_remaining),
        "as_of": as_of_dt.strftime("%Y-%m-%d"),
        "month_end": m_end.strftime("%Y-%m-%d"),
    }
