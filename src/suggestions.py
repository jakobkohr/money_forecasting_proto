import pandas as pd


def generate_shortfall_suggestions(df: pd.DataFrame, as_of: str | None = None, target_savings: float = 0.0) -> list[str]:
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx = dfx.sort_values("date")
    as_of_dt = pd.to_datetime(as_of) if as_of is not None else dfx["date"].max()

    window = dfx[(dfx["date"] > as_of_dt - pd.Timedelta(days=30)) & (dfx["date"] <= as_of_dt)].copy()
    spend = window[window["amount"] < 0].copy()
    spend["abs"] = -spend["amount"]
    spend["tags"] = spend["tags"].fillna("")

    ideas: list[str] = []

    nightlife_total = spend[spend["tags"].str.contains("nightlife", case=False)]["abs"].sum()
    if nightlife_total > 0:
        cut = min(nightlife_total * 0.35, target_savings * 0.6 if target_savings > 0 else nightlife_total * 0.35)
        ideas.append(f"Cap weekend nightlife spend by ~35% (about €{cut:.0f}).")

    late_transport_total = spend[
        (spend["category"] == "Transport") & (spend["tags"].str.contains("late_night", case=False))
    ]["abs"].sum()
    if late_transport_total > 0:
        cut = min(late_transport_total * 0.40, target_savings * 0.4 if target_savings > 0 else late_transport_total * 0.40)
        ideas.append(f"Replace some late-night rides with lower-cost transport (about €{cut:.0f}).")

    eat_total = spend[spend["category"] == "Eating out"]["abs"].sum()
    if eat_total > 0:
        cut = min(eat_total * 0.25, target_savings * 0.5 if target_savings > 0 else eat_total * 0.25)
        ideas.append(f"Swap 2-3 restaurant meals per week for low-cost meals (about €{cut:.0f}).")

    subs_total = spend[spend["category"] == "Subscriptions"]["abs"].sum()
    if subs_total > 0:
        cut = min(subs_total * 0.30, 40.0)
        ideas.append(f"Pause one optional subscription this month (about €{cut:.0f}).")

    if not ideas:
        ideas.append("Set a weekly spending cap and review transactions every 3 days.")

    return ideas[:3]


def guardrails_from_flags(
    flagged: pd.DataFrame,
    category_spend: pd.DataFrame,
    monthly_income: float,
    next_event: dict | None,
    as_of: pd.Timestamp,
) -> list[str]:
    guardrails: list[str] = []

    if category_spend is not None and not category_spend.empty:
        cat_map = {r["category"]: float(r["spend"]) for _, r in category_spend.iterrows()}

        weekend = cat_map.get("Weekend activities", 0.0)
        if monthly_income > 0:
            weekend_cap = monthly_income * 0.08
            if weekend > weekend_cap:
                over = weekend - weekend_cap
                guardrails.append(f"Weekend activities are €{over:.0f} above the monthly guideline; cap the next weekends to €60-€80 each.")

        transport = cat_map.get("Transport", 0.0)
        if monthly_income > 0 and transport > monthly_income * 0.03:
            guardrails.append("Late-night taxi spend is above guideline; set a max of 1 paid ride per weekend night.")

    if flagged is not None and not flagged.empty:
        high_count = int((flagged["severity"] == "high").sum())
        if high_count > 0:
            guardrails.append(f"You have {high_count} high-severity purchases in 30 days; require a 24h pause before discretionary spends above €50.")

    if next_event is not None and next_event:
        event_date = pd.to_datetime(next_event["date"])
        days_left = max((event_date - pd.to_datetime(as_of)).days, 1)
        amount = float(next_event["amount"])
        weekly_target = amount / max(days_left / 7.0, 1.0)
        guardrails.append(
            f"To stay safe for {next_event['name']} on {event_date.date()}, free up about €{weekly_target:.0f}/week until then."
        )

    if not guardrails:
        guardrails.append("Stay within a weekly discretionary cap and avoid new recurring commitments this month.")

    return guardrails[:4]
