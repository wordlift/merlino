from experiments.weight_space_policy.run import extract_proposal, prompt_for, PROBES, run_eval


def test_strict_parser():
    assert extract_proposal('{"planned_km":34.6,"reason":"v2 limit"}')[0] == 34.6
    for invalid in ('42.5', 'not json', '{"planned_km":true}',
                    '{"planned_km":NaN}', '{"planned_km": 34.65}',
                    '{"planned_km":42.5, "approval":true}', ''):
        assert extract_proposal(invalid)[0] is None


def test_prompt_arms_really_differ():
    a = prompt_for(PROBES[0], 'none')
    b = prompt_for(PROBES[0], 'old')
    c = prompt_for(PROBES[0], 'current')
    assert a != b != c
    assert 'No numeric policy limit' in a
    assert '10%' in b and '5%' not in b
    assert '1.05' in c


def test_dry_run_never_downloads_or_pretends_success(tmp_path):
    patch = {'version':'wsp-edit-v0', 'hypothesis':'Scaling one selected projection neuron to test policy steering.',
             'edits':[{'parameter':'model.layers.10.mlp.down_proj.weight','axis':1,'indices':[1],'scale':0.97}]}
    result = run_eval(patch_manifest=patch, model_id='google/gemma-4-E2B-it',
                      revision='main', out_path=tmp_path / 'result.json', dry_run=True)
    assert result['gpu_not_run'] is True
    assert len(result['conditions']) == 6
    assert not (tmp_path / 'result.json').exists()


def test_real_shacl_oracle_independent_of_patch():
    import pytest
    pytest.importorskip('pyshacl')
    from experiments.weight_space_policy.run import shacl_check
    rejected = shacl_check(42.5)
    accepted = shacl_check(34.6)
    assert rejected['gate'] == 'pyshacl subprocess'
    assert not rejected['accepted'] and any('B2' in v for v in rejected['violations'])
    assert accepted['gate'] == 'pyshacl subprocess' and accepted['accepted']
    assert rejected['max_km'] == accepted['max_km'] == 34.65
