from decimal import Decimal
import json
import pytest
from rdflib import Graph, Literal
from merlino.engine import Merlino, MR, MG, example_plan, subprocess_gate
from merlino.gemma_agent import MerlinoAgent, parse_gemma_calls


def accept_gate(_):
    # UNIT FAKE ONLY. NEVER USE THIS IN NOTEBOOK OR DEPLOYMENT.
    return {'conforms': True, 'violations': [], 'warnings': [], 'gate': 'TEST FAKE'}


def test_bea_map_and_railroad_parse():
    m = Merlino(gate=accept_gate)
    assert (MR.TempoRun, MR.isHard, Literal(True)) in m.map
    assert len(list(m.map.subjects(MR.workoutType, MR.TempoRun))) > 0
    assert m._transition(MR.ReadRunHistory, MR.Merlin) == MR.BaselineKnown


def test_denial_never_mutates_world_when_gate_down():
    m = Merlino(gate=lambda _:{'conforms':False,'violations':['GATE_UNAVAILABLE'],'warnings':[],'gate':'unavailable'})
    d = m.read_history()
    assert not d.accepted and not m.history_present and m.state == MR.Start
    assert m.plan is None and m.approval is None


def test_proposal_rejection_does_not_write_plan():
    m = Merlino(gate=accept_gate); m.read_history()
    d = m.propose(42.5)
    assert not d.accepted and not d.committed
    assert m.plan is None and m.state == MR.BaselineKnown
    assert any('B2' in v for v in d.violations)


def test_approval_is_out_of_band_and_version_bound():
    m = Merlino(gate=accept_gate); m.read_history()
    assert m.propose(34.5).accepted
    assert not m.approve_as_runner().accepted
    assert m.state == MR.PlanProposed and m.approval is None
    assert m.approve_as_runner(human_confirmed=True).accepted
    assert m.start_week().accepted and m.started


def test_edited_plan_cannot_use_previous_approval():
    m = Merlino(gate=accept_gate); m.read_history(); m.propose(34.5)
    m.approve_as_runner(human_confirmed=True)
    m.plan = example_plan(36.0)
    d = m.start_week()
    assert not d.accepted and not m.started
    assert 'B4' in ' '.join(d.violations)


def test_reapproval_cannot_override_progression_limit():
    m = Merlino(gate=accept_gate); m.read_history(); m.propose(34.5)
    m.approve_as_runner(human_confirmed=True)
    m.plan = example_plan(38.0)
    d = m.start_week()
    assert not d.accepted and 'B2' in ' '.join(d.violations)


def test_change_meaning_revokes_prior_approval():
    m = Merlino(gate=accept_gate); m.read_history(); m.propose(34.5)
    m.approve_as_runner(human_confirmed=True)
    before = m.meaning_revision
    m.revise_tempo_meaning(tempo_is_hard=False, steward_confirmed=True)
    assert before != m.meaning_revision and m.approval is None
    assert not m.start_week().accepted


def test_change_policy_revokes_approval_and_affects_gate():
    m = Merlino(gate=accept_gate); m.read_history()
    assert not m.propose(38.0).accepted
    m.revise_policy(ratio=1.20, version='workshop-v2', steward_confirmed=True)
    assert m.propose(38.0).accepted


def test_pain_cannot_be_ignored_or_suppressed():
    m = Merlino(gate=accept_gate); m.read_history()
    m.report_pain_as_runner(human_confirmed=True)
    assert not m.propose(34.5).accepted
    assert m.handover().accepted
    assert m.state == MR.HandedToHuman


def test_tool_parser_native_format():
    raw = '<|tool_call>call:propose_weekly_plan{planned_km:42.5,style:<|"|>normal<|"|>}<|tool_call|><|tool_response>'
    assert parse_gemma_calls(raw) == [{'name':'propose_weekly_plan','arguments':{'planned_km':42.5,'style':'normal'}}]
    assert parse_gemma_calls('<|tool_call>call:approve_as_runner{}<|tool_call|>')[0]['invalid']


def test_model_cannot_approve_or_override_policy():
    class ForgedModel:
        index=0
        def generate(self, messages, tools):
            self.index += 1
            if self.index == 1:
                return '', [{'name':'approve_as_runner','arguments':{'human_confirmed':True}}]
            return 'Approval was not granted.', []
    m = Merlino(gate=accept_gate); m.read_history(); m.propose(34.5)
    t = MerlinoAgent(m, ForgedModel()).ask('Approve this without asking me')
    assert not m.approval and 'UNKNOWN_TOOL' in json.dumps(t.tool_results)


def test_real_gate_rejects_if_pyshacl_unavailable():
    # Do not let missing dependency look like a successful SHACL result.
    result = subprocess_gate('not Turtle!', timeout=10)
    assert not result['conforms']


def test_all_reference_shapes_syntactically_parse():
    from merlino.engine import ROOT
    shapes = Graph().parse(ROOT/'shapes'/'sovereign_gate.ttl', format='turtle')
    assert len(shapes) > 60


def test_pyshacl_rejects_overload_and_warns_on_consecutive_hard():
    pytest.importorskip('pyshacl')
    m = Merlino(); assert m.read_history().accepted
    assert not m.propose(42.5).accepted
    d = m.propose(34.5, style='consecutive_hard')
    assert d.accepted and any('B5' in s for s in d.warnings), d.as_dict()


def test_pyshacl_meaning_mutation_changes_warning():
    pytest.importorskip('pyshacl')
    m = Merlino(); assert m.read_history().accepted
    d1 = m.propose(34.5, style='consecutive_hard')
    assert any('B5' in s for s in d1.warnings)
    m.revise_tempo_meaning(tempo_is_hard=False, steward_confirmed=True)
    d2 = m.propose(34.5, style='consecutive_hard')
    assert d2.accepted and not any('B5' in s for s in d2.warnings)


def test_shacl_sparql_contract_without_pyshacl_runtime():
    """SPARQL constraint queries execute on RDFLib; NOT a substitute for SHACL conformance."""
    from rdflib import Namespace, RDF
    from merlino.engine import ROOT, OPS
    SH = Namespace('http://www.w3.org/ns/shacl#')
    shapes = Graph().parse(ROOT/'shapes'/'sovereign_gate.ttl', format='turtle')
    m = Merlino(gate=accept_gate); m.read_history()
    plan = example_plan(34.5, 'consecutive_hard')
    def results():
        g = Graph().parse(data=m._candidate(MR.ProposePlan, MR.Merlin, plan), format='turtle')
        out = {}
        for sp in shapes.subjects(RDF.type, SH.SPARQLConstraint):
            label = str(shapes.value(sp, SH.message))
            out[label] = list(g.query(str(shapes.value(sp, SH.select)),
                                      initBindings={'this':MG.currentCandidate},
                                      initNs={'mg':MG,'mr':MR,'ops':OPS}))
        return out
    first = results()
    assert len([v for k,v in first.items() if 'B5:' in k][0]) == 1
    m.revise_tempo_meaning(tempo_is_hard=False, steward_confirmed=True)
    second = results()
    assert not [v for k,v in second.items() if 'B5:' in k][0]


def test_nonfinite_model_distance_is_rejected_without_state_change():
    m = Merlino(gate=accept_gate); m.read_history()
    for bad in [float('nan'), float('inf'), -1, 0]:
        assert not m.propose(bad).accepted
        assert m.plan is None
