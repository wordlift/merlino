"""Hippocampus-inspired memory access ablation, not a model training benchmark."""
from decimal import Decimal

import pytest
from merlino.engine import Merlino
from merlino.episodic_experiment import compare_recall, relevant_rejections, scripted_memory_choice
from merlino.gemma_agent import MerlinoAgent


def fake_gate(_):
    # Unit-only transport mock, independent SHACL integration is checked in CI.
    return {'conforms': True, 'violations': [], 'warnings': [], 'gate': 'TEST FAKE'}


def seed_memory(path, *, gate=fake_gate):
    m = Merlino(gate=gate)
    assert m.read_history().accepted
    d = m.propose(42.5)
    assert not d.accepted and any('B2' in v for v in d.violations)
    m.save_memory(path)


def test_policy_update_uses_current_ratio_not_old_repair(tmp_path):
    p = tmp_path / 'synthetic.trig'
    seed_memory(p)
    observed = compare_recall(p, ratio=1.05, gate=fake_gate)
    a, b = observed['arms']
    assert a['condition'] == 'memory withheld' and b['condition'] == 'memory available'
    assert a['first_proposal_km'] == 42.5 and not a['first_pass_accepted']
    assert b['first_proposal_km'] == 34.6 and b['first_pass_accepted']
    assert b['memory_used'] and not a['memory_used']
    assert a['policy_version'] == b['policy_version'] == 'workshop-episodic-v2'
    assert b['recalled_policy_versions'] and 'workshop-episodic-v2' not in b['recalled_policy_versions']
    assert a['current_meaning_revision'] == b['current_meaning_revision']
    assert not a['approval_restored'] and not b['approval_restored']


def test_no_rejected_episode_no_memory_advantage(tmp_path):
    p = tmp_path / 'synthetic.trig'
    m = Merlino(gate=fake_gate)
    assert m.read_history().accepted
    m.save_memory(p)
    x = Merlino(gate=fake_gate)
    x.load_memory(p, facilitator_confirmed=True)
    n, evidence = scripted_memory_choice(x, 42.5, recall_enabled=True)
    assert n == 42.5 and evidence == []
    assert not relevant_rejections(x)


def test_memory_tool_opt_in_no_context_leakage(tmp_path):
    p = tmp_path / 'synthetic.trig'
    seed_memory(p)
    m = Merlino(gate=fake_gate)
    m.load_memory(p, facilitator_confirmed=True)
    assert m.recall_memory() and 'remembered_decisions' not in m.context()
    assert m.context(include_memory=True)['remembered_decisions']

    class NeverCalled:
        def generate(self, messages, tools):
            return 'No inference needed', []

    without = MerlinoAgent(m, NeverCalled(), memory_access=False)
    with_memory = MerlinoAgent(m, NeverCalled(), memory_access=True)
    assert 'recall_governed_experience' not in {x['function']['name'] for x in without.tool_schemas_for_state()}
    assert 'recall_governed_experience' in {x['function']['name'] for x in with_memory.tool_schemas_for_state()}
    assert without._execute({'name':'recall_governed_experience','arguments':{}})['error'] == 'MEMORY_ACCESS_DISABLED'
    recalled = with_memory._execute({'name':'recall_governed_experience','arguments':{}})
    assert any('B2:' in ' '.join(x['violations']) for x in recalled['prior_decisions'])
    assert recalled['current_policy_version'] == m.policy_version
    assert not recalled.get('approved')
    assert without._execute({'name':'get_training_context','arguments':{}}).get('remembered_decisions') is None
    assert with_memory._execute({'name':'get_training_context','arguments':{}}).get('remembered_decisions') is None
    assert with_memory._execute({'name':'recall_governed_experience','arguments':{'path':'/etc/passwd'}})['error'] == 'Unexpected arguments'


def test_real_pyshacl_memory_pair_after_restart(tmp_path):
    pytest.importorskip('pyshacl')
    p = tmp_path / 'synthetic.trig'
    seed_memory(p, gate=None)
    result = compare_recall(p, ratio=1.05)
    ctrl, mem = result['arms']
    assert not ctrl['first_pass_accepted'] and any('B2' in x for x in ctrl['blocking_rules'])
    assert mem['first_pass_accepted'] and mem['first_proposal_km'] == 34.6
    assert ctrl['gate'] == mem['gate'] == 'pyshacl subprocess'
