import pandas as pd
import numpy as np

def transactions_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx["amount"] = pd.to_numeric(dfx["amount"])

    # Daily totals
    daily = (
        dfx.groupby(dfx["date"].dt.date)
        .agg(
            spend_total=("amount", lambda s: float(np.abs(s[s < 0]).sum())),
            income_total=("amount", lambda s: float(s[s > 0].sum())),
        )
        .reset_index()
        .rename(columns={"date": "day"})
    )
    daily["day"] = pd.to_datetime(daily["day"])
    daily = daily.sort_values("day").reset_index(drop=True)

    # Build a running balance from daily net (optional; useful for UI)
    daily["net"] = daily["income_total"] - daily["spend_total"]
    daily["balance"] = daily["net"].cumsum()

    return daily

def add_calendar_features(daily: pd.DataFrame, payday_day: int = 1) -> pd.DataFrame:
    out = daily.copy()
    out["weekday"] = out["day"].dt.weekday
    out["is_weekend"] = (out["weekday"] >= 4).astype(int)  # Fri-Sun
    out["day_of_month"] = out["day"].dt.day
    out["month"] = out["day"].dt.month
    out["year"] = out["day"].dt.year

    # days since payday (assume payday is fixed day-of-month)
    # if day_of_month >= payday_day -> since payday = day_of_month - payday_day
    # else -> since payday = day_of_month + days_in_prev_month - payday_day
    days_in_month = out["day"].dt.days_in_month
    out["days_since_payday"] = np.where(
        out["day_of_month"] >= payday_day,
        out["day_of_month"] - payday_day,
        out["day_of_month"] + (out["day"] - pd.offsets.MonthBegin(1)).dt.days_in_month - payday_day,
    )

    # days to month end
    month_end = out["day"] + pd.offsets.MonthEnd(0)
    out["days_to_month_end"] = (month_end - out["day"]).dt.days

    return out

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Rolling stats on spend_total (shifted so we don't leak today's spend into features)
    s = out["spend_total"].shift(1)

    out["roll7_mean_spend"] = s.rolling(7, min_periods=3).mean()
    out["roll14_mean_spend"] = s.rolling(14, min_periods=5).mean()
    out["roll7_std_spend"] = s.rolling(7, min_periods=3).std()

    # Previous day spend
    out["prev_spend"] = out["spend_total"].shift(1)
    out["roll3_mean_spend"] = s.rolling(3, min_periods=2).mean()
    out["roll14_weekend_rate"] = out["is_weekend"].shift(1).rolling(14, min_periods=7).mean()

    return out

def build_ml_table(transactions_df: pd.DataFrame, payday_day: int = 1) -> pd.DataFrame:
    daily = transactions_to_daily(transactions_df)
    daily = add_calendar_features(daily, payday_day=payday_day)
    daily = add_rolling_features(daily)

    # Label: next day spend (what model predicts)
    daily["y_next_spend"] = daily["spend_total"].shift(-1)

    # Drop rows where we don't have enough history or label
    daily = daily.dropna().reset_index(drop=True)

    return daily

def get_feature_columns() -> list[str]:
    return [
        "weekday",
        "is_weekend",
        "day_of_month",
        "days_since_payday",
        "days_to_month_end",
        "roll7_mean_spend",
        "roll14_mean_spend",
        "roll7_std_spend",
        "prev_spend",
    ]
