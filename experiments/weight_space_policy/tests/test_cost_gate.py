import pytest
from experiments.weight_space_policy.cost_gate import CostEstimate, approve_preflight

def test_requires_explicit_authorization():
    with pytest.raises(PermissionError):
        approve_preflight(CostEstimate(0.000542, 2, 16, 3600), remaining_usd=10)

def test_fails_when_budget_exceeded():
    with pytest.raises(RuntimeError):
        approve_preflight(CostEstimate(0.000694, 2, 32, 36000), remaining_usd=10, explicitly_authorized=True)

def test_valid_small_estimate():
    assert 0 < approve_preflight(CostEstimate(0.000542, 2, 16, 600), remaining_usd=10, explicitly_authorized=True) < 10
