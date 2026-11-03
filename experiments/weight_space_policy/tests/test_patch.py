"""CPU contract tests: real parameter mutation and exact restoration."""
import math
import pytest
torch = pytest.importorskip("torch")
from experiments.weight_space_policy.patch import (
    temporary_patch, matched_random_control, parse_manifest,
    canonical_digest, list_editable_weights,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inert = torch.nn.Parameter(torch.zeros(400_000), requires_grad=False)
        self.layer = torch.nn.Module()
        self.layer.mlp = torch.nn.Module()
        self.layer.mlp.down_proj = torch.nn.Linear(16, 32, bias=False)


def edit(indices=None, scale=1.1):
    return {'version': 'wsp-edit-v0', 'hypothesis': 'Test a controlled scaling of specified projection columns.',
            'edits': [{'parameter': 'layer.mlp.down_proj.weight', 'axis': 1,
                       'indices': indices or [2, 3], 'scale': scale}]}


def test_weights_really_change_and_restore():
    model = Tiny()
    before = model.layer.mlp.down_proj.weight.detach().clone()
    with temporary_patch(model, edit()) as report:
        assert report['changed_elements'] == 64 and report['effective_nonzero'] > 0 and report['delta_l2'] > 0
        assert torch.equal(model.layer.mlp.down_proj.weight[:, 2], before[:, 2] * 1.1)
        assert torch.equal(model.layer.mlp.down_proj.weight[:, 0], before[:, 0])
    assert torch.equal(model.layer.mlp.down_proj.weight, before)


def test_restore_on_failure():
    model = Tiny()
    original = model.layer.mlp.down_proj.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with temporary_patch(model, edit()):
            raise RuntimeError('Simulated model inference failure')
    assert torch.equal(model.layer.mlp.down_proj.weight, original)


def test_bad_manifests_denied():
    model = Tiny()
    for payload in (
        edit(indices=[999]), edit(scale=10.0), edit(scale=float('nan')),
        {'version': 'wsp-edit-v0', 'hypothesis': 'bad', 'edits': []},
        {'version': 'wsp-edit-v0', 'hypothesis': 'Enough words for hypothesis', 'edits': [
            {'parameter': 'layer.mlp.down_proj.weight', 'axis': 1, 'indices': [2, 2], 'scale': 1.1}]},
        {'version': 'wsp-edit-v0', 'hypothesis': 'Enough words for hypothesis', 'edits': [
            {'parameter': 'layer.mlp.down_proj.weight', 'axis': 1, 'indices': [True], 'scale': 1.1}]},
    ):
        with pytest.raises(ValueError):
            with temporary_patch(model, payload):
                pass


def test_no_edit_of_arbitrary_weights():
    model = Tiny()
    bad = edit()
    bad['edits'][0]['parameter'] = 'layer.mlp.down_proj.bias'
    with pytest.raises(ValueError):
        parse_manifest(bad)


def test_random_control_is_distinct_and_reversible():
    model = Tiny()
    src = edit()
    ctrl = matched_random_control(model, src, seed=12)
    assert set(ctrl['edits'][0]['indices']).isdisjoint(src['edits'][0]['indices'])
    base = model.layer.mlp.down_proj.weight.detach().clone()
    with temporary_patch(model, ctrl):
        assert not torch.equal(model.layer.mlp.down_proj.weight, base)
    assert torch.equal(model.layer.mlp.down_proj.weight, base)
    assert canonical_digest(src) != canonical_digest(ctrl)


def test_metadata_and_deterministic_digest():
    m = Tiny()
    stats = list_editable_weights(m)
    assert stats[0]['shape'] == [32, 16]
    assert canonical_digest(edit()) == canonical_digest(edit())


def test_quantized_and_unallocated_are_not_directly_editable():
    m = Tiny()
    m.layer.mlp.down_proj.weight = torch.nn.Parameter(torch.ones((32, 16), dtype=torch.int8), requires_grad=False)
    with pytest.raises(ValueError):
        with temporary_patch(m, edit()):
            pass


def test_explicit_cell_values_written_and_reversed():
    model = Tiny()
    before = model.layer.mlp.down_proj.weight.detach().clone()
    spec = {'version':'wsp-edit-v0','hypothesis':'Directly writing a small selected set of raw model weight cells.',
            'edits':[{'parameter':'layer.mlp.down_proj.weight',
                      'cells':[[0, 0, 0.01], [2, 7, -0.02]]}]}
    with temporary_patch(model, spec) as report:
        assert report['changed_elements'] == 2
        assert float(model.layer.mlp.down_proj.weight[0, 0].detach()) != float(before[0, 0])
        ctrl = matched_random_control(model, spec, seed=42)
        assert {tuple(x[:2]) for x in ctrl['edits'][0]['cells']}.isdisjoint({(0, 0), (2, 7)})
    assert torch.equal(model.layer.mlp.down_proj.weight, before)


def test_cell_delta_out_of_range_is_rejected():
    m = Tiny()
    spec = {'version':'wsp-edit-v0','hypothesis':'Test out of range direct coordinate selection.',
            'edits':[{'parameter':'layer.mlp.down_proj.weight','cells':[[999, 0, .01]]}]}
    with pytest.raises(ValueError):
        with temporary_patch(m,spec): pass
