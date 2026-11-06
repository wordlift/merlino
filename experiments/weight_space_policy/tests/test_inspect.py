import pytest
torch = pytest.importorskip('torch')
from experiments.weight_space_policy.inspect import weight_window


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.mlp = torch.nn.Module()
        self.layer.mlp.down_proj = torch.nn.Linear(5, 6, bias=False)


def test_bounded_raw_values():
    t = Toy()
    meta = weight_window(t, parameter='layer.mlp.down_proj.weight', row=1, col=2, height=2, width=2)
    assert len(meta['values']) == 2
    assert meta['values'][0][0] == round(float(t.layer.mlp.down_proj.weight[1, 2].detach()), 8)


def test_reject_unbounded_or_wrong_parameter():
    t = Toy()
    for arg in [dict(row=0,col=0,height=100,width=1),
                dict(row=5,col=0,height=2,width=1),
                dict(row=0,col=0,height=1,width=1,parameter='embed_tokens.weight')]:
        kwargs=dict(parameter='layer.mlp.down_proj.weight',row=0,col=0,height=1,width=1)
        kwargs.update(arg)
        with pytest.raises(ValueError): weight_window(t,**kwargs)
