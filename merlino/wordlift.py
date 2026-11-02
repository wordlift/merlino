"""Explicit opt-in Resolve call with synthetic-only payloads.

Do not put the workshop key into the notebook, Git or shared outputs.
"""
from __future__ import annotations

import requests

SYNTHETIC_TEXT = "Tempo run and easy run are workout types."
INLINE_VOCABULARY = {"entities": [
    {"id": "https://example.org/merlino/v1#TempoRun", "name": "Tempo run", "types": ["Workout"],
     "description": "A running workout sustained around a challenging but controlled pace."},
    {"id": "https://example.org/merlino/v1#EasyRun", "name": "Easy run", "types": ["Workout"],
     "description": "A running workout at conversational effort."},
]}


def synthetic_resolve_payload() -> dict:
    """Return exactly what will cross the network boundary."""
    return {"text": SYNTHETIC_TEXT, "language": "en", "dataset_uri": "inline",
            "dataset": INLINE_VOCABULARY, "include": ["candidates", "evidence"]}


def resolve_synthetic_only(api_key: str, *, session=None) -> dict:
    if not api_key or not api_key.strip():
        raise ValueError("WordLift key required")
    client = session or requests.Session()
    headers = {"Authorization": "Key " + api_key.strip()}
    me = client.get("https://api.wordlift.io/accounts/me", headers=headers, timeout=20)
    me.raise_for_status()
    account_id = me.json()["accountId"]
    # The caller cannot pass arbitrary text or RDF to this demo function.
    response = client.post(
        f"https://api.wordlift.io/accounts/{account_id}/resolve",
        headers=headers, json=synthetic_resolve_payload(), timeout=40)
    response.raise_for_status()
    result = response.json()
    return {"backend": "WordLift Resolve (remote, explicit opt-in)",
            "dataset_uri": result.get("dataset_uri"),
            "mentions": [{"text": m.get("text"), "status": m.get("status"),
                          "iri": (m.get("entity") or {}).get("id"), "reason": m.get("reason")}
                         for m in result.get("mentions", [])]}
