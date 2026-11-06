"""Bounded raw weight peek to support a human/LLM-authored numeric patch.

This exports only a small rectangular window of a selected editable projection.
The purpose is raw-weight inspection (not activation training or gradient fitting).
"""
from __future__ import annotations

from .patch import ELIGIBLE_SUFFIXES


def weight_window(model, *, parameter: str, row: int, col: int,
                  height: int = 8, width: int = 8) -> dict:
    if not parameter.endswith(ELIGIBLE_SUFFIXES):
        raise ValueError('Only MLP projection tensors are inspectable')
    if any(type(v) is not int for v in (row, col, height, width)):
        raise ValueError('Coordinates must be integers')
    if row < 0 or col < 0 or not (1 <= height <= 32) or not (1 <= width <= 32):
        raise ValueError('Inspection window is limited to 32x32 values')
    found = dict(model.named_parameters()).get(parameter)
    if found is None or found.ndim != 2 or not found.dtype.is_floating_point:
        raise ValueError('Parameter not present as a floating 2-D projection')
    if row + height > found.shape[0] or col + width > found.shape[1]:
        raise ValueError('Selected inspection window exceeds tensor dimensions')
    window = found.detach()[row:row+height, col:col+width].float().cpu()
    return {'parameter': parameter, 'row_start': row, 'col_start': col,
            'height': height, 'width': width,
            'values': [[round(float(x), 8) for x in y] for y in window.tolist()]}
