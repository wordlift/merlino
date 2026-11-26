"""Single-target research specification for Gemma 4 E2B."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    name: str = "gemma"
    model_id: str = "google/gemma-4-E2B-it"
    architecture: str = "multimodal"
    editable_with_v0: bool = True

def get_spec(name: str) -> ModelSpec:
    if name != "gemma":
        raise ValueError(f"Unsupported research model: {name}")
    return ModelSpec()

def require_editable(name: str) -> ModelSpec:
    return get_spec(name)

def plan_models(names=("gemma",)) -> list[dict]:
    return [vars(get_spec(name)).copy() for name in names]
