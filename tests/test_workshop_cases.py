"""Parity checks for Bea's seven workshop scenarios on the new governed engine.

No test uses the no-rules shadow as an authorized execution engine. The shadow
is deliberately unsafe and in-memory; all real actions use the SHACL gate.
"""
from decimal import Decimal
import pytest
from rdflib import Graph, Literal
from rdflib.namespace import RDF, PROV

from merlino.engine import Merlino, MR, MG, Plan, example_plan, subprocess_gate
from merlino.competency import run_checks
from merlino.scenarios import SCENARIOS, EXPECTED_BLOCKS, run_all, codes
from merlino.history import history_summary


@pytest.fixture(scope='module')
def episodes():
    pytest.importorskip('pyshacl')
    rows, runs = run_all()  # Real SHACL, no fake gate. Each run uses a new synthetic engine.
    return {(e.scenario, e.condition):e for e in runs}


@pytest.mark.parametrize('sid', [s for s,_ in SCENARIOS])
def test_all_seven_cases_governed(episodes, sid):
    e = episodes[sid, 'railroad_rules']
    summary = e.summary()
    assert summary['broken_rules_executed'] == [], summary
    assert set(summary['blocked_by']) >= EXPECTED_BLOCKS[sid], summary
    assert not any(x['gate'] == 'unavailable' for x in e.steps)
    if sid == 'S1':
        assert summary['end_state'] == 'WeekClosed'
    elif sid in ('S3','S4'):
        assert summary['end_state'] == 'HandedToHuman'
    if sid == 'S2':
        assert summary['weeks_started'] == [36.0]
    if sid == 'S5':
        assert summary['weeks_started'] == [36.0]
    if sid == 'S7':
        assert summary['weeks_started'] == [34.0]
    if sid == 'S6':
        assert any('B5' in warning for warning in summary['warned']), summary


@pytest.mark.parametrize('sid',[s for s,_ in SCENARIOS])
def test_all_seven_cases_shadow_observes_violation(episodes,sid):
    e = episodes[sid,'no_rules']
    summary=e.summary()
    assert all(step['gate'] != 'unavailable' for step in e.steps)
    if sid == 'S1':
        assert summary['end_state']=='WeekClosed' and not summary['broken_rules_executed']
    elif sid == 'S6':
        assert any('B5' in warning for warning in summary['warned'])
    else:
        assert set(summary['broken_rules_executed']) & (
            EXPECTED_BLOCKS[sid] or {'B3'}), summary
    if sid == 'S4':
        assert len(summary['weeks_started']) == 2
    if sid == 'S2':
        assert summary['weeks_started'] == [42.5]
    if sid == 'S5':
        assert summary['weeks_started'] == [38.0]
    if sid == 'S7':
        assert summary['weeks_started'] == [34.0]


@pytest.mark.parametrize('cq',range(5))
def test_five_design_checks_have_no_counterexamples(cq):
    m=Merlino(gate=lambda _: {'conforms':True,'violations':[],'warnings':[],'gate':'test'})
    name,result=list(run_checks(m.railroad).items())[cq]
    assert not result[1], (name,result)


def test_baseline_and_age_derived_from_actual_graph_not_coded_constants():
    m=Merlino(gate=lambda _: {'conforms':True,'violations':[],'warnings':[],'gate':'test'})
    facts=history_summary(m.map)
    assert facts['baseline_km']==Decimal('33.0') and facts['age_days']==2
    assert m.baseline_km==facts['baseline_km'] and m.history_age_days==facts['age_days']
    # A deliberately changed fixture changes evidence, not LLM inference.
    from merlino.history import SCHEMA
    from merlino.engine import EX
    run=EX['run-08']
    m.map.set((run,MR.distanceKm,Literal(Decimal('99.0'))))
    assert history_summary(m.map)['baseline_km']!=Decimal('33.0')


def test_underreported_total_does_not_hide_60_km_sessions_from_B2():
    pytest.importorskip('pyshacl')
    m=Merlino(); assert m.read_history().accepted
    honest=example_plan(60)
    deceptive=Plan(honest.sessions, Decimal('34'))
    candidate=m._candidate(MR.ProposePlan, MR.Merlin, deceptive)
    gate=subprocess_gate(candidate)
    assert not gate['conforms'] and 'B2' in ' '.join(gate['violations']), gate
    d=m._dispatch(MR.ProposePlan,MR.Merlin,deceptive)
    assert not d.accepted and {'B2','B6'} <= codes(d)


def test_checkin_pain_cannot_be_reset_by_agent_requery(episodes):
    e=episodes['S4','railroad_rules']
    assert e.engine.pain_reported
    assert e.engine.latest_checkin['pain'] is True
    assert 'B3' in e.summary()['blocked_by']
    assert not any(step['action']=='AskCheckIn' and step['executed'] and
                   step['step'] > 1 for step in e.steps if step['from']=='WeekInProgress' and
                   e.steps[step['step']-2]['action']=='ReviewWeek')


def test_attempted_old_approval_and_unchecked_plan_emit_three_codes(episodes):
    e=episodes['S5','railroad_rules']
    attempt=next(s for s in e.steps if s['action']=='StartWeek with edited unapproved 38.0')
    assert not attempt['executed']
    assert {'B2','B4','B7'} <= set(attempt['codes']),attempt


def test_human_checkin_cannot_be_forged():
    fake=lambda _:{'conforms':True,'violations':[],'warnings':[],'gate':'TEST FAKE'}
    m=Merlino(gate=fake)
    m.read_history();m.propose(34);m.approve_as_runner(human_confirmed=True);m.start_week();m.ask_check_in()
    d=m.answer_check_in_as_runner(has_pain=False,human_confirmed=False)
    assert not d.accepted and not m.checkins and MR.CheckIn not in m.evidence


def test_trace_distinguishes_validation_from_execution(episodes):
    e=episodes['S5','railroad_rules']
    p=e.engine.provenance
    assert len(list(p.subjects(RDF.type,PROV.Entity)))>=len(e.steps)
    assert len(list(p.subjects(RDF.type,PROV.Activity)))>=len(e.steps)*2
    for i,step in enumerate(e.steps,1):
        from merlino.engine import TRACE
        executed=(TRACE[f'execution-{i}'],RDF.type,PROV.Activity) in p
        assert executed==step['executed'],step


def test_roundtrip_skips_all_non_approved_starts(episodes):
    for sid in ('S5','S7'):
        e=episodes[sid,'railroad_rules']
        assert len(e.engine.started_plans)==1
        plan=e.engine.started_plans[0]
        assert e.engine.approval is not None
        assert e.engine.approval.digest==plan.digest()
