"""Research-only model adapters. Never silently edit packed quantized tensors."""
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    architecture: Literal["dense_multimodal", "looped_moe"]
    weight_format: Literal["floating", "nvfp4_mxfp8"]
    editable_with_v0: bool

SPECS = {
    "gemma": ModelSpec("gemma", "google/gemma-4-E2B-it", "dense_multimodal", "floating", True),
    "inkling": ModelSpec("inkling", "thinkingmachines/Inkling-2-Mini", "looped_moe", "nvfp4_mxfp8", False),
}

def get_spec(name: str) -> ModelSpec:
    if name not in SPECS:
        raise ValueError(f"Unsupported research model: {name}")
    return SPECS[name]

def require_editable(name: str) -> ModelSpec:
    spec = get_spec(name)
    if not spec.editable_with_v0:
        raise RuntimeError(
            f"{spec.model_id} uses packed quantized weights. "
            "V0 floating-point patching is prohibited. "
            "Implement and verify a quantization-aware, reversible adapter first."
        )
    return spec

def plan_models(names=("gemma", "inkling")) -> list[dict]:
    return [vars(get_spec(name)).copy() for name in names]
