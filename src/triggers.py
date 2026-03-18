from __future__ import annotations

import pandas as pd

from src.analytics import CATEGORY_LIMITS


def detect_triggers(
    tx: pd.DataFrame,
    monthly_income: float,
    category_30: pd.DataFrame,
    flagged: pd.DataFrame,
    projection: dict,
    all_events: pd.DataFrame,
    as_of: pd.Timestamp,
) -> list[dict]:
    """
    Analyze the user's financial data and return a ranked list of trigger dicts.
    Each trigger has:
      - query: a natural-language string describing the financial situation (used as RAG query)
      - priority: int (higher = more urgent)
    Only the highest-priority trigger is used to generate RAG advice.
    """
    triggers: list[dict] = []

    cat_map: dict[str, float] = (
        {r["category"]: float(r["spend"]) for _, r in category_30.iterrows()}
        if not category_30.empty
        else {}
    )

    projected_end = projection["projected_end_balance"]
    current_balance = projection["current_balance"]

    # --- Trigger 1: Projected negative balance (highest urgency) ---
    if projected_end < 0:
        shortfall = abs(projected_end)
        triggers.append(
            {
                "query": (
                    f"user is projected to go negative by €{shortfall:.0f} at month end. "
                    f"Current balance is €{current_balance:.0f}. "
                    "They need to cut spending urgently to avoid overdraft fees and financial stress."
                ),
                "priority": 10,
            }
        )

    # --- Trigger 2: Very low projected balance (at risk) ---
    elif projected_end < 200:
        triggers.append(
            {
                "query": (
                    f"user projected to have only €{projected_end:.0f} left at month end. "
                    f"Current balance is €{current_balance:.0f}. "
                    "Very thin buffer — unexpected expenses would cause overdraft. "
                    "They need to reduce discretionary spending to build a safety cushion."
                ),
                "priority": 8,
            }
        )

    # --- Trigger 3: Upcoming obligation not covered ---
    if projection.get("event_projection"):
        for ep in sorted(projection["event_projection"], key=lambda x: x["date"]):
            if ep["projected_balance"] < ep["amount"] * 0.5:
                triggers.append(
                    {
                        "query": (
                            f"user has an upcoming obligation '{ep['name']}' of €{ep['amount']:.0f} "
                            f"but projected balance at that date is only €{ep['projected_balance']:.0f}. "
                            "They risk not being able to cover this payment and need to cut spending now."
                        ),
                        "priority": 9,
                    }
                )
                break  # Report only the most urgent upcoming event

    # --- Trigger 4: Category overspending ---
    worst_category: tuple[str, float, float] | None = None
    worst_excess_pct = 0.0

    for category, limit in CATEGORY_LIMITS.items():
        cat_spend = cat_map.get(category, 0.0)
        cap = monthly_income * limit if monthly_income > 0 else 0.0
        if cap > 0 and cat_spend > cap * 1.10:  # 10% over limit threshold
            excess_pct = (cat_spend - cap) / cap
            if excess_pct > worst_excess_pct:
                worst_excess_pct = excess_pct
                worst_category = (category, cat_spend, cap)

    if worst_category is not None:
        category, cat_spend, cap = worst_category
        pct_of_income = (cat_spend / monthly_income) * 100 if monthly_income > 0 else 0
        excess = cat_spend - cap
        triggers.append(
            {
                "query": (
                    f"user overspending on {category.lower()}: spent €{cat_spend:.0f} in last 30 days "
                    f"({pct_of_income:.0f}% of income), but guideline cap is €{cap:.0f} "
                    f"({CATEGORY_LIMITS[category]*100:.0f}% of income). "
                    f"€{excess:.0f} over budget. They need targeted advice to cut this category."
                ),
                "priority": 7,
            }
        )

    # --- Trigger 5: Multiple high-severity impulse purchases ---
    if not flagged.empty:
        high_flags = flagged[flagged["severity"] == "high"].copy()
        if len(high_flags) >= 2:
            total_flagged = float(high_flags["amount"].abs().sum())
            top_cat = high_flags["category"].value_counts().index[0]
            triggers.append(
                {
                    "query": (
                        f"user has {len(high_flags)} high-severity flagged impulse or overspend purchases "
                        f"totalling €{total_flagged:.0f} in the last 30 days, "
                        f"mostly in {top_cat}. "
                        "This pattern of impulse buying is eroding their budget."
                    ),
                    "priority": 6,
                }
            )
        elif len(high_flags) == 1:
            row = high_flags.iloc[0]
            triggers.append(
                {
                    "query": (
                        f"user made a high-severity impulse purchase at {row['merchant']} "
                        f"for €{abs(float(row['amount'])):.0f} in {row['category']}. "
                        "This single purchase exceeds healthy spending thresholds."
                    ),
                    "priority": 5,
                }
            )

    # --- Trigger 6: Total spend very high relative to income (fallback) ---
    total_30d_spend = sum(cat_map.values())
    if monthly_income > 0 and total_30d_spend > monthly_income * 0.80:
        pct = (total_30d_spend / monthly_income) * 100
        triggers.append(
            {
                "query": (
                    f"user is spending {pct:.0f}% of monthly income (€{total_30d_spend:.0f} of €{monthly_income:.0f}) "
                    "in discretionary categories over the last 30 days, "
                    "leaving almost nothing for savings or unexpected costs."
                ),
                "priority": 4,
            }
        )

    # Sort by priority descending and return
    triggers.sort(key=lambda x: x["priority"], reverse=True)
    return triggers
