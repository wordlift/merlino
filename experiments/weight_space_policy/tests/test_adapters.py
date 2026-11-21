"""No GPU needed: prohibit unsafe Inkling edits until implemented."""
import pytest
from experiments.weight_space_policy.adapters import get_spec, require_editable, plan_models

def test_both_models_registered():
    assert [x["name"] for x in plan_models()] == ["gemma", "inkling"]

def test_gemma_existing_edit_path():
    assert require_editable("gemma").editable_with_v0

def test_inkling_quantized_patch_fails_closed():
    with pytest.raises(RuntimeError, match="quantization-aware"):
        require_editable("inkling")

def test_unknown_model_fails_closed():
    with pytest.raises(ValueError):
        get_spec("unknown")
