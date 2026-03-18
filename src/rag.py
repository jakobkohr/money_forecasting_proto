from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

TIPS_PATH = Path(__file__).parent.parent / "data" / "financial_tips.json"
EMBED_MODEL = "embed-english-v3.0"
CHAT_MODEL = "command-r-08-2024"
TOP_K = 4


def load_tips() -> list[dict]:
    """Load the financial tips knowledge base from disk."""
    with open(TIPS_PATH, "r") as f:
        return json.load(f)


def embed_knowledge_base(co: Any, tips: list[dict]) -> np.ndarray:
    """
    Pre-embed all tips using Cohere embed API.
    Returns an (n_tips, embedding_dim) float32 array.
    Each tip is embedded as '<title>. <body>'.
    """
    texts = [f"{t['title']}. {t['body']}" for t in tips]
    resp = co.embed(
        texts=texts,
        model=EMBED_MODEL,
        input_type="search_document",
        embedding_types=["float"],
    )
    return np.array(resp.embeddings.float_, dtype=np.float32)


def _cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix of document vectors.
    Returns a 1-D array of similarities (one per document).
    """
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    row_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9
    d_norm = doc_matrix / row_norms
    return d_norm @ q_norm


def retrieve_tips(
    co: Any,
    query: str,
    tips: list[dict],
    tip_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed the trigger query and retrieve the top-k most relevant tips
    via cosine similarity search over the pre-embedded knowledge base.
    """
    resp = co.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    query_vec = np.array(resp.embeddings.float_[0], dtype=np.float32)
    similarities = _cosine_similarity(query_vec, tip_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [tips[i] for i in top_indices]


def _build_prompt(trigger_query: str, user_context: dict, retrieved_tips: list[dict]) -> tuple[str, str]:
    """
    Build the system preamble and user message for the Cohere chat call.
    Returns (system_prompt, user_message).
    """
    tips_block = "\n\n".join(
        f"TIP {i + 1} — {t['title']}:\n{t['body']}"
        for i, t in enumerate(retrieved_tips)
    )

    ctx = user_context
    cat_overspends = ctx.get("category_overspends", "None identified")
    flagged_summary = ctx.get("flagged_summary", "None")

    context_block = (
        f"Monthly income: €{ctx.get('monthly_income', 0):,.0f}\n"
        f"Current balance: €{ctx.get('current_balance', 0):,.2f}\n"
        f"Projected end-of-month balance: €{ctx.get('projected_end_balance', 0):,.2f}\n"
        f"Habit spend forecast (remaining days): €{ctx.get('habit_spend_forecast', 0):,.0f}\n"
        f"Upcoming commitments this month: €{ctx.get('upcoming_commitments', 0):,.0f}\n"
        f"Risk state: {ctx.get('risk_state', 'Unknown')}\n"
        f"Category overspends (last 30 days): {cat_overspends}\n"
        f"Flagged purchases (last 30 days): {flagged_summary}"
    )

    system_prompt = (
        "You are a personal finance advisor. You give concise, specific, and actionable advice "
        "grounded in the user's real financial numbers. "
        "Always reference the actual euro amounts from the user's data. "
        "Be direct and practical — no generic platitudes. "
        "You must respond with valid JSON only, exactly matching the requested schema."
    )

    user_message = (
        f"USER FINANCIAL SNAPSHOT:\n{context_block}\n\n"
        f"MAIN CONCERN DETECTED:\n{trigger_query}\n\n"
        f"RELEVANT FINANCIAL TIPS (use these as grounding material):\n{tips_block}\n\n"
        "Based on the above, generate exactly 3 personalized, actionable advice cards. "
        "Each card must directly reference the user's specific numbers (balances, category amounts, etc.). "
        "Draw from the tips provided but adapt them to fit the user's exact situation.\n\n"
        "Respond with a JSON object in this exact format — no extra text before or after:\n"
        "{\n"
        '  "advice": [\n'
        "    {\n"
        '      "title": "Short specific title (max 8 words)",\n'
        '      "action": "One concrete action the user should take this week",\n'
        '      "estimated_savings": 60,\n'
        '      "priority": "high",\n'
        '      "reason": "One sentence explaining why this matters for this user specifically, with their actual numbers"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- priority must be one of: high, medium, low\n"
        "- estimated_savings is monthly savings in euros (integer, realistic)\n"
        "- Generate exactly 3 advice items\n"
        "- Return only valid JSON, no markdown, no extra text"
    )

    return system_prompt, user_message


def _parse_advice(raw: str) -> list[dict]:
    """
    Parse the LLM's JSON response into a list of advice dicts.
    Handles cases where the model wraps the JSON in markdown code blocks.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try to extract a JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return []

    try:
        parsed = json.loads(match.group())
        advice = parsed.get("advice", [])
        # Validate each item has required fields
        valid = []
        for item in advice:
            if all(k in item for k in ("title", "action", "estimated_savings", "priority")):
                valid.append(item)
        return valid
    except (json.JSONDecodeError, AttributeError):
        return []


def get_advice(
    co: Any,
    trigger_query: str,
    user_context: dict,
    tips: list[dict],
    tip_embeddings: np.ndarray,
) -> list[dict]:
    """
    Full RAG pipeline:
      1. Retrieve top-k relevant tips via cosine similarity
      2. Build augmented prompt with user context + retrieved tips
      3. Call Cohere chat for grounded generation
      4. Parse and return structured advice cards

    Returns a list of advice dicts, each with:
      title, action, estimated_savings, priority, reason
    """
    retrieved = retrieve_tips(co, trigger_query, tips, tip_embeddings, top_k=TOP_K)
    system_prompt, user_message = _build_prompt(trigger_query, user_context, retrieved)

    response = co.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw_text = response.message.content[0].text.strip()
    return _parse_advice(raw_text)


def build_user_context(
    monthly_income: float,
    current_balance: float,
    projected_end_balance: float,
    habit_spend_forecast: float,
    upcoming_commitments: float,
    risk_state: str,
    category_30: Any,  # pd.DataFrame
    flagged: Any,  # pd.DataFrame
    monthly_income_ref: float,
) -> dict:
    """
    Build a structured user context dict for the RAG prompt from app-level variables.
    """
    from src.analytics import CATEGORY_LIMITS

    # Summarise category overspends
    cat_overspend_parts = []
    if not category_30.empty:
        cat_map = {r["category"]: float(r["spend"]) for _, r in category_30.iterrows()}
        for category, limit in CATEGORY_LIMITS.items():
            spend = cat_map.get(category, 0.0)
            cap = monthly_income_ref * limit if monthly_income_ref > 0 else 0.0
            if cap > 0 and spend > cap:
                cat_overspend_parts.append(
                    f"{category} €{spend:.0f} (cap €{cap:.0f})"
                )
    category_overspends = ", ".join(cat_overspend_parts) if cat_overspend_parts else "None"

    # Summarise flagged purchases
    flagged_summary = "None"
    if not flagged.empty:
        high = flagged[flagged["severity"] == "high"]
        med = flagged[flagged["severity"] == "medium"]
        parts = []
        if len(high) > 0:
            parts.append(f"{len(high)} high-severity")
        if len(med) > 0:
            parts.append(f"{len(med)} medium-severity")
        if parts:
            total = float(flagged["amount"].abs().sum())
            flagged_summary = f"{', '.join(parts)} flagged purchases totalling €{total:.0f}"

    return {
        "monthly_income": monthly_income,
        "current_balance": current_balance,
        "projected_end_balance": projected_end_balance,
        "habit_spend_forecast": habit_spend_forecast,
        "upcoming_commitments": upcoming_commitments,
        "risk_state": risk_state,
        "category_overspends": category_overspends,
        "flagged_summary": flagged_summary,
    }
