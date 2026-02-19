import pandas as pd
import numpy as np

def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds recurring_detected (0/1) using simple rules:
    - same merchant+category appears >= 3 times
    - interval roughly weekly (7±2) or monthly (30±5)
    - amount similarity (median absolute deviation small)
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")

    out["recurring_detected"] = 0

    grouped = out.groupby(["merchant", "category"], dropna=False)
    for (merchant, category), g in grouped:
        if len(g) < 3:
            continue

        # day gaps
        gaps = g["date"].diff().dt.days.dropna()
        if gaps.empty:
            continue

        gap_med = float(np.median(gaps))
        weeklyish = 5 <= gap_med <= 9
        monthlyish = 25 <= gap_med <= 35

        # amount stability
        amounts = g["amount"].to_numpy()
        med = float(np.median(amounts))
        mad = float(np.median(np.abs(amounts - med)))  # median abs deviation

        # for small subscriptions, mad may be tiny already; allow proportion-based threshold
        stable = mad <= max(1.0, 0.12 * abs(med))

        if (weeklyish or monthlyish) and stable:
            out.loc[g.index, "recurring_detected"] = 1

    # restore string dates for display
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out
