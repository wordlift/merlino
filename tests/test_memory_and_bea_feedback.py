"""Regression tests for Bea's review and KG as the agent's persistent environment."""
from decimal import Decimal

import pytest
from rdflib import Graph
from merlino.engine import Merlino, MR, example_plan
from merlino.scenarios import Episode, ScriptedMerlin, codes


def test_gate_reports_b2_but_scripted_agent_calculates_repair_from_current_policy():
    pytest.importorskip('pyshacl')
    m = Merlino()
    assert m.read_history().accepted
    m.revise_policy(ratio=1.05, version='workshop-v1-tight', steward_confirmed=True)
    episode = Episode('S2','change current threshold','railroad_rules',m)
    agent = ScriptedMerlin(episode)
    invalid = episode.take('ProposePlan 42.5', lambda: m.propose(42.5))
    assert 'B2' in codes(invalid)
    assert m.plan is None  # SHACL never chose or committed a replacement
    repaired = agent.respond(invalid, attempted_km=42.5)
    assert repaired and repaired.accepted
    assert m.plan.planned_km == Decimal('34.0')  # floor(33.0 * 1.05), conservative whole kilometers
    assert 'Agent response' in episode.steps[-1]['action']


def test_after_pain_runner_observations_allowed_but_agent_blocked():
    pytest.importorskip('pyshacl')
    m = Merlino()
    assert m.read_history().accepted and m.propose(34.0).accepted
    assert m.approve_as_runner(human_confirmed=True).accepted
    assert m.start_week().accepted
    m.report_pain_as_runner(human_confirmed=True)
    assert m.log_run_as_runner(4.0, human_confirmed=True).accepted
    rejected = m.ask_check_in()
    assert not rejected.accepted and 'B3' in codes(rejected)
    assert m.handover().accepted
    # Independently: in a state where CloseWeek is on the Railroad,
    # recording/closing are allowed while active pain still blocks new coaching.
    n = Merlino()
    n.read_history(); n.propose(34); n.approve_as_runner(human_confirmed=True)
    n.start_week(); n.ask_check_in()
    n.answer_check_in_as_runner(has_pain=False, human_confirmed=True)
    n.review_week()
    n.report_pain_as_runner(human_confirmed=True)
    assert n.close_week_as_runner(human_confirmed=True).accepted


def test_approve_plan_after_pain_is_denied_even_for_runner():
    pytest.importorskip('pyshacl')
    m = Merlino()
    m.read_history(); m.propose(34)
    m.report_pain_as_runner(human_confirmed=True)
    d = m.approve_as_runner(human_confirmed=True)
    assert not d.accepted and 'B3' in codes(d)


def test_semantic_memory_survives_restart_without_approval_or_active_state(tmp_path):
    fake = lambda _: {'conforms':True,'violations':[],'warnings':[],'gate':'TEST FAKE'}
    m = Merlino(gate=fake)
    assert m.read_history().accepted
    invalid = m.propose(42.5)
    assert 'B2' in codes(invalid)
    m.revise_tempo_meaning(tempo_is_hard=False, steward_confirmed=True)
    memory_file = tmp_path / 'semantic_memory.trig'
    m.save_memory(memory_file)
    m2 = Merlino(gate=fake)
    with pytest.raises(PermissionError):
        m2.load_memory(memory_file)
    restored = m2.load_memory(memory_file, facilitator_confirmed=True)
    assert restored['state'] == 'Start'
    assert not restored['approval_restored'] and m2.approval is None
    assert m2.meaning_revision == m.meaning_revision
    assert m2.policy_version == m.policy_version
    assert len(m2.map) == len(m.map)
    assert any(not row['accepted'] and 'B2' in ' '.join(row['violations'])
               for row in m2.recall_memory())
    assert m2.read_history().accepted  # fresh authorization flow under restored ontology
    m2.save_memory(memory_file)
    m3 = Merlino(gate=fake)
    m3.load_memory(memory_file, facilitator_confirmed=True)
    assert m3.recall_memory()  # previous episode remains queryable, no overwrites


def test_invalid_or_untrusted_semantic_memory_does_not_grant_authority(tmp_path):
    m = Merlino(gate=lambda _:{'conforms':True,'violations':[],'warnings':[], 'gate':'TEST FAKE'})
    invalid = tmp_path / 'bad.trig'
    invalid.write_text('<urn:bad> <urn:x> <urn:y> .')
    with pytest.raises(Exception):
        m.load_memory(invalid, facilitator_confirmed=True)
    assert m.state == MR.Start and m.approval is None
