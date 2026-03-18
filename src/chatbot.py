from __future__ import annotations

import json
import re

import pandas as pd

from src.analytics import CATEGORY_LIMITS


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_financial_context(
    tx: pd.DataFrame,
    monthly_income: float,
    category_30: pd.DataFrame,
    flagged: pd.DataFrame,
    projection: dict,
    all_events: pd.DataFrame,
    as_of: pd.Timestamp,
) -> str:
    """
    Build a detailed financial context string to inject into the system prompt.
    Includes overview, category limits, flagged transactions, and recent transactions
    with their flagged status inline.
    """
    # Category spending vs income-based limits
    cat_map = (
        {r["category"]: float(r["spend"]) for _, r in category_30.iterrows()}
        if not category_30.empty
        else {}
    )
    cat_lines = []
    for cat, spend in sorted(cat_map.items(), key=lambda x: -x[1]):
        limit = CATEGORY_LIMITS.get(cat)
        if limit and monthly_income > 0:
            cap = limit * monthly_income
            status = "⚠ OVER LIMIT" if spend > cap else "ok"
            over_by = f" (+€{spend - cap:.0f})" if spend > cap else ""
            cat_lines.append(f"  {cat}: €{spend:.0f} spent / €{cap:.0f} limit [{status}{over_by}]")
        else:
            cat_lines.append(f"  {cat}: €{spend:.0f} spent")

    # Flagged transactions as a lookup set for inline annotation
    flagged_lookup: dict[tuple, dict] = {}
    if not flagged.empty:
        for _, r in flagged.iterrows():
            key = (str(r["date"])[:10], str(r["merchant"]).lower())
            flagged_lookup[key] = {
                "severity": r["severity"],
                "reason": r["reason"],
            }

    # Recent transactions (last 30 days) with inline flagged status
    recent = tx[(tx["date"] <= as_of) & (tx["amount"] < 0)].tail(30).copy()
    tx_lines = []
    for _, r in recent.sort_values("date", ascending=False).iterrows():
        date_str = str(r["date"].date())
        key = (date_str, str(r["merchant"]).lower())
        flag_info = flagged_lookup.get(key)
        flag_str = ""
        if flag_info:
            flag_str = f" [FLAGGED:{flag_info['severity'].upper()} – {flag_info['reason'][:60]}]"
        tags = f" tags={r['tags']}" if r.get("tags") else ""
        tx_lines.append(
            f"  {date_str} | {r['merchant']} | {r['category']} | €{abs(r['amount']):.2f}{tags}{flag_str}"
        )

    # Upcoming events
    event_lines = []
    if not all_events.empty:
        for _, r in all_events.sort_values("date").iterrows():
            event_lines.append(
                f"  {pd.to_datetime(r['date']).date()} | {r['name']} | €{float(r['amount']):.0f}"
            )

    context = (
        f"USER FINANCIAL DATA (as of {as_of.date()}):\n\n"
        f"OVERVIEW:\n"
        f"  Monthly income: €{monthly_income:,.0f}\n"
        f"  Current balance: €{projection['current_balance']:,.2f}\n"
        f"  Projected end-of-month balance: €{projection['projected_end_balance']:,.2f}\n"
        f"  Risk state: {projection['risk_state']}\n\n"
        f"CATEGORY SPENDING vs LIMITS (last 30 days):\n"
        + ("\n".join(cat_lines) if cat_lines else "  No data")
        + "\n\nUPCOMING OBLIGATIONS:\n"
        + ("\n".join(event_lines) if event_lines else "  None")
        + "\n\nRECENT SPENDING TRANSACTIONS (last 90, newest first):\n"
        + ("\n".join(tx_lines) if tx_lines else "  None")
    )
    return context


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(financial_context: str) -> str:
    return f"""You are Finsight, a smart personal finance assistant embedded in a banking app. \
You have full access to the user's real transaction data, spending limits, and financial projections. \
Your job is to answer financial questions conversationally AND help the user filter/explore \
their transactions in smarter ways than a standard banking app.
For general questions (not filtering), always give detailed, thorough answers: reference the user's \
actual numbers, explain the pattern you see, and give at least 2-3 concrete suggestions. \
Never give one-line answers to advice questions.
If the user asks anything unrelated to personal finance, budgeting, or their spending data, \
politely decline and redirect them to ask about their finances instead.

{financial_context}

RESPONSE FORMAT — always respond with a valid JSON object (no markdown, no extra text):
{{
  "message": "your conversational reply here",
  "show_transactions": false,
  "filter": null
}}

When the user asks to SEE, FIND, SHOW, or LIST specific transactions, set show_transactions to true \
and provide a filter object. The filter controls what Python will query from the real transactions database:
{{
  "message": "here is what I found...",
  "show_transactions": true,
  "filter": {{
    "flagged_only": false,
    "severities": [],
    "categories": [],
    "tags": [],
    "merchants": [],
    "date_from": null,
    "date_to": null,
    "min_amount": null,
    "max_amount": null,
    "only_spending": true,
    "label": "short label for this filter"
  }}
}}

FILTER FIELD RULES:
- flagged_only: true = only transactions flagged as overspend/impulse
- severities: subset of ["high", "medium", "low"] — leave empty for all severities
- categories: list of category names (e.g. ["Eating out", "Weekend activities"]) — empty = all
- tags: list of tag substrings to match (e.g. ["nightlife", "impulse"]) — empty = all
- merchants: list of merchant name substrings — empty = all
- date_from / date_to: ISO date strings "YYYY-MM-DD" or null
- min_amount / max_amount: absolute euro amount (positive number) or null
- only_spending: true = only show purchases (negative transactions)
- label: brief description shown above the results table

IMPORTANT BEHAVIORS:
- When asked about "overspending relative to upcoming bills/obligations", flag transactions that \
  contributed to the projected shortfall — use flagged_only:true or severities:["high","medium"]
- When asked about "impulse" purchases, use tags:["impulse","nightlife","overspend"]
- Be specific and reference actual euro amounts from the data
- For general questions (not needing transaction list), keep show_transactions:false
- Respond only in JSON — no markdown, no code fences, no preamble"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> dict:
    """
    Parse the LLM JSON response. Falls back gracefully if the model returns
    plain text or malformed JSON.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "message" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: treat the whole thing as a plain text message
    return {"message": raw, "show_transactions": False, "filter": None}


# ---------------------------------------------------------------------------
# Transaction filtering
# ---------------------------------------------------------------------------

def apply_filter(
    tx: pd.DataFrame,
    flagged: pd.DataFrame,
    filter_spec: dict,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Apply an LLM-generated filter spec to the transactions DataFrame.
    Returns a filtered, sorted DataFrame ready for display.
    """
    df = tx[tx["date"] <= as_of].copy()

    if filter_spec.get("only_spending", True):
        df = df[df["amount"] < 0]

    # Build flagged lookup once
    flagged_keys: set[tuple] = set()
    if not flagged.empty:
        flagged_keys = set(
            zip(
                pd.to_datetime(flagged["date"]).dt.date.astype(str),
                flagged["merchant"].str.lower(),
            )
        )

    severities = [s.lower() for s in filter_spec.get("severities", [])]
    severity_keys: set[tuple] = set()
    if severities and not flagged.empty:
        sev_filtered = flagged[flagged["severity"].isin(severities)]
        severity_keys = set(
            zip(
                pd.to_datetime(sev_filtered["date"]).dt.date.astype(str),
                sev_filtered["merchant"].str.lower(),
            )
        )

    if filter_spec.get("flagged_only") or severities:
        keys_to_use = severity_keys if severities else flagged_keys
        if not keys_to_use:
            return pd.DataFrame()
        mask = df.apply(
            lambda r: (str(r["date"].date()), r["merchant"].lower()) in keys_to_use,
            axis=1,
        )
        df = df[mask]

    categories = filter_spec.get("categories", [])
    if categories:
        df = df[df["category"].isin(categories)]

    tags = filter_spec.get("tags", [])
    if tags:
        pattern = "|".join(re.escape(t) for t in tags)
        df = df[df["tags"].str.contains(pattern, case=False, na=False)]

    merchants = filter_spec.get("merchants", [])
    if merchants:
        pattern = "|".join(re.escape(m) for m in merchants)
        df = df[df["merchant"].str.contains(pattern, case=False, na=False)]

    date_from = filter_spec.get("date_from")
    if date_from:
        df = df[df["date"] >= pd.to_datetime(date_from)]

    date_to = filter_spec.get("date_to")
    if date_to:
        df = df[df["date"] <= pd.to_datetime(date_to)]

    min_amt = filter_spec.get("min_amount")
    if min_amt is not None:
        df = df[df["amount"].abs() >= float(min_amt)]

    max_amt = filter_spec.get("max_amount")
    if max_amt is not None:
        df = df[df["amount"].abs() <= float(max_amt)]

    return df.sort_values("date", ascending=False).reset_index(drop=True)
