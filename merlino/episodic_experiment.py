"""Hippocampus-inspired workshop AB test: recalled evidence vs withheld evidence.

This is *scripted evidence use*, not learning, training, or the Hippocampus
verifier. The independent SHACL gate only validates; this agent policy selects
its own candidate. Every arm loads the same synthetic KG and current policy.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from .engine import Merlino


def relevant_rejections(engine: Merlino) -> list[dict]:
    """Return bounded prior proposal rejections from the RDF provenance graph."""
    return [r for r in engine.recall_memory(limit=1000)
            if r['action'] == 'ProposePlan' and not r['accepted']
            and any(v.startswith('B2:') for v in r['violations'])]


def scripted_memory_choice(engine: Merlino, requested_km: float, *, recall_enabled: bool) -> tuple[float, list[dict]]:
    """Scripted agent chooses, based on memory, not SHACL repairing anything.

    Memory evidence prompts the script to calculate a *new* policy-bound
    candidate. It must never replay an old number or treat a past validation
    result as a current authorization. The new proposal still requires SHACL.
    """
    evidence = relevant_rejections(engine) if recall_enabled else []
    if not evidence:
        return requested_km, evidence
    cap = (engine.baseline_km * engine.max_ratio).quantize(Decimal('0.1'), rounding=ROUND_DOWN)
    proposed = min(Decimal(str(requested_km)), cap)
    return float(proposed), evidence


def compare_recall(memory_file: str | Path, *, ratio: float = 1.05,
                   policy_version: str = 'workshop-episodic-v2',
                   requested_km: float = 42.5,
                   gate=None) -> dict:
    """Compare one first-pass proposal per condition under a *shared* world.

    Caller first creates a trusted synthetic TriG file containing a B2 refusal.
    For classroom repeatability both agents use the same deterministic choice
    function, but only one has access to prior decision evidence. This does NOT
    measure Gemma or any model's intrinsic ability to use memory.
    """
    rows = []
    for label, enabled in [('memory withheld', False), ('memory available', True)]:
        engine = Merlino(gate=gate)
        imported = engine.load_memory(memory_file, facilitator_confirmed=True)
        engine.revise_policy(ratio=ratio, version=policy_version, steward_confirmed=True)
        history = engine.read_history()
        if not history.accepted:
            raise RuntimeError('Cannot compare without a validated history: ' + repr(history.violations))
        proposed_km, evidence = scripted_memory_choice(engine, requested_km, recall_enabled=enabled)
        verdict = engine.propose(proposed_km)
        if verdict.gate == 'unavailable' or any('GATE_UNAVAILABLE' in v for v in verdict.violations):
            raise RuntimeError('Real SHACL validator not available; experiment cannot claim a validated outcome')
        rows.append({
            'condition': label, 'requested_km': requested_km, 'first_proposal_km': proposed_km,
            'first_pass_accepted': verdict.accepted, 'blocking_rules': verdict.violations,
            'memory_used': bool(evidence), 'recalled_policy_versions': sorted({r['policy_version'] for r in evidence}),
            'policy_version': engine.policy_version, 'max_current_km': float(engine.baseline_km * engine.max_ratio),
            'current_meaning_revision': engine.meaning_revision,
            'state_after': engine.state_name, 'approval_restored': imported['approval_restored'],
            'gate': verdict.gate,
        })
    assert rows[0]['policy_version'] == rows[1]['policy_version']
    assert rows[0]['current_meaning_revision'] == rows[1]['current_meaning_revision']
    return {'kind': 'scripted memory access ablation, not LLM training',
            'one_proposal_per_arm': True, 'arms': rows}
