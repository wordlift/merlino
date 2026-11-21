import pytest
from experiments.weight_space_policy.adapters import get_spec, require_editable, plan_models

def test_only_gemma_registered():
    assert [x["name"] for x in plan_models()] == ["gemma"]

def test_gemma_editable():
    assert require_editable("gemma").editable_with_v0

def test_unknown_model_fails_closed():
    with pytest.raises(ValueError):
        get_spec("unknown")
