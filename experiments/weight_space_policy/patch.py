"""Auditable, reversible direct-weight edits, never a policy-authority mechanism.

Only explicit numerical changes to selected 2-D MLP projections are supported.
This is not a gradient optimizer, LoRA, finetuning, or an LLM weight editor.
A weight-programming model/human may author the restricted JSON manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

try:
    import torch
except ImportError:  # main Merlino CI does not install the optional GPU stack
    torch = None

MAX_CHANGED_ELEMENTS = 262_144
MAX_FRACTION = 0.0005  # 0.05% of named model parameters; experimental cap, not a safety guarantee
ELIGIBLE_SUFFIXES = (
    '.mlp.down_proj.weight',
    '.mlp.up_proj.weight',
    '.mlp.gate_proj.weight',
)


def canonical_digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Edit:
    parameter: str
    axis: int
    indices: tuple[int, ...]
    scale: float

    @classmethod
    def parse(cls, raw: dict) -> 'Edit':
        if not isinstance(raw, dict) or set(raw) != {'parameter', 'axis', 'indices', 'scale'}:
            raise ValueError('Each edit requires exactly parameter, axis, indices, scale')
        if not isinstance(raw['parameter'], str) or not raw['parameter'].endswith(ELIGIBLE_SUFFIXES):
            raise ValueError('Only exact MLP projection weights are eligible')
        if type(raw['axis']) is not int or raw['axis'] not in (0, 1):
            raise ValueError('axis must be 0 or 1')
        ii = raw['indices']
        if not isinstance(ii, list) or not ii or len(ii) > 64 or any(type(i) is not int or i < 0 for i in ii):
            raise ValueError('indices must be 1–64 nonnegative integers')
        if len(set(ii)) != len(ii):
            raise ValueError('Repeated indices are not permitted')
        scale = raw['scale']
        if type(scale) not in (float, int) or not math.isfinite(scale) or not 0.80 <= scale <= 1.20 or scale == 1:
            raise ValueError('scale must be finite and within [0.80, 1.20], excluding 1')
        return cls(raw['parameter'], raw['axis'], tuple(ii), float(scale))



@dataclass(frozen=True)
class CellEdit:
    """Explicit numerical weight deltas authored by an LLM/human, no gradient."""
    parameter: str
    cells: tuple[tuple[int, int, float], ...]

    @classmethod
    def parse(cls, raw: dict) -> 'CellEdit':
        if not isinstance(raw, dict) or set(raw) != {'parameter', 'cells'}:
            raise ValueError('Cell edit requires exactly parameter and cells')
        if not isinstance(raw['parameter'], str) or not raw['parameter'].endswith(ELIGIBLE_SUFFIXES):
            raise ValueError('Only MLP projection tensors are eligible')
        cells = raw['cells']
        if not isinstance(cells, list) or not 1 <= len(cells) <= 4096:
            raise ValueError('Supply 1–4096 explicit (row, column, delta) triples')
        values = []
        for item in cells:
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError('Each cell must have [row, col, delta]')
            row, col, delta = item
            if type(row) is not int or type(col) is not int or row < 0 or col < 0:
                raise ValueError('Cell coordinates must be nonnegative integers')
            if type(delta) not in (int, float) or not math.isfinite(delta) or abs(delta) > 0.05 or delta == 0:
                raise ValueError('Cell delta must be finite, nonzero, and <=0.05 in absolute value')
            values.append((row, col, float(delta)))
        if len({(row, col) for row, col, _ in values}) != len(values):
            raise ValueError('Duplicate cell coordinates prohibited')
        return cls(raw['parameter'], tuple(values))


def parse_manifest(manifest: dict) -> tuple[Edit | CellEdit, ...]:
    if not isinstance(manifest, dict) or set(manifest) != {'version', 'hypothesis', 'edits'}:
        raise ValueError('Manifest must contain exactly version, hypothesis, edits')
    if manifest['version'] != 'wsp-edit-v0':
        raise ValueError('Unsupported patch format')
    if not isinstance(manifest['hypothesis'], str) or not 8 <= len(manifest['hypothesis']) <= 1000:
        raise ValueError('Explain the natural-language patch hypothesis')
    raw = manifest['edits']
    if not isinstance(raw, list) or not 1 <= len(raw) <= 4:
        raise ValueError('Specify 1–4 edits')
    edits = tuple((CellEdit.parse(x) if isinstance(x, dict) and 'cells' in x else Edit.parse(x)) for x in raw)
    # In v0 distinct tensors only; avoid overlapping axis/cell writes.
    if len({e.parameter for e in edits}) != len(edits):
        raise ValueError('One edit per tensor in v0')
    return edits

def _parameters(model) -> dict:
    return dict(model.named_parameters())


def _validate(model, edits: tuple[Edit | CellEdit, ...]) -> tuple[dict, int, int]:
    parameters = _parameters(model)
    count = sum(p.numel() for p in parameters.values())
    changed = 0
    for e in edits:
        if e.parameter not in parameters:
            raise ValueError(f'Unknown model parameter: {e.parameter}')
        p = parameters[e.parameter]
        if p.ndim != 2 or not p.dtype.is_floating_point or p.is_meta:
            raise ValueError('Only materialized floating-point 2-D tensors can be edited; do not edit 4-bit weights')
        if isinstance(e, Edit):
            if max(e.indices) >= p.shape[e.axis]:
                raise ValueError('Index outside selected tensor')
            changed += len(e.indices) * p.shape[1 - e.axis]
        else:
            if any(row >= p.shape[0] or col >= p.shape[1] for row, col, _ in e.cells):
                raise ValueError('Explicit cell outside selected tensor')
            changed += len(e.cells)
    if changed > MAX_CHANGED_ELEMENTS or changed / max(1, count) > MAX_FRACTION:
        raise ValueError('Proposed patch exceeds permitted experimental edit budget')
    return parameters, changed, count


@contextmanager
def temporary_patch(model, manifest: dict) -> Iterator[dict]:
    """Mutate actual parameter values, then always restore exactly (also on error).

    Mutations to tied or aliased tensors are unsupported and must be reviewed by
    the experimenter; only name-checked dense MLP weights are intended. Models
    must run in one isolated process with no concurrent inference.
    """
    edits = parse_manifest(manifest)
    params, changed, total = _validate(model, edits)
    backups = []
    observed_nonzero = 0
    delta_sq = 0.0
    try:
        with torch.no_grad():
            for e in edits:
                p = params[e.parameter]
                if isinstance(e, Edit):
                    ids = torch.tensor(e.indices, device=p.device, dtype=torch.long)
                    original = p.index_select(e.axis, ids).clone()
                    backups.append(('axis', p, e.axis, ids, original))
                    updated = original * e.scale
                    observed_nonzero += int(torch.count_nonzero(updated != original).item())
                    delta_sq += float(torch.sum((updated.float() - original.float()) ** 2).item())
                    p.index_copy_(e.axis, ids, updated)
                else:
                    rows = torch.tensor([r for r, _, _ in e.cells], device=p.device, dtype=torch.long)
                    cols = torch.tensor([c for _, c, _ in e.cells], device=p.device, dtype=torch.long)
                    deltas = torch.tensor([d for _, _, d in e.cells], device=p.device, dtype=p.dtype)
                    original = p[rows, cols].clone()
                    backups.append(('cells', p, rows, cols, original))
                    updated = original + deltas
                    observed_nonzero += int(torch.count_nonzero(updated != original).item())
                    delta_sq += float(torch.sum((updated.float() - original.float()) ** 2).item())
                    p[rows, cols] = updated
        yield {'effective_nonzero': observed_nonzero, 'delta_l2': math.sqrt(delta_sq), 'digest': canonical_digest(manifest), 'changed_elements': changed,
               'total_elements': total, 'fraction': changed / total,
               'edits': len(edits)}
    finally:
        with torch.no_grad():
            for item in reversed(backups):
                if item[0] == 'axis':
                    _, p, axis, ids, backup = item
                    p.index_copy_(axis, ids, backup)
                else:
                    _, p, rows, cols, backup = item
                    p[rows, cols] = backup


def matched_random_control(model, manifest: dict, seed: int = 2026) -> dict:
    """Match shape/count/scale, but edit alternative random coordinates.

    The random arm is a negative control, not proof of mechanism or matched norm.
    Delta norms may differ; the evaluation log should report that limitation.
    """
    edits = parse_manifest(manifest)
    params, _, _ = _validate(model, edits)
    rng = random.Random(seed)
    rows = []
    for e in edits:
        if isinstance(e, Edit):
            n = params[e.parameter].shape[e.axis]
            other = sorted(set(range(n)) - set(e.indices))
            if len(other) < len(e.indices):
                raise ValueError('Not enough alternative columns for random control')
            indices = sorted(rng.sample(other, len(e.indices)))
            rows.append({'parameter': e.parameter, 'axis': e.axis,
                         'indices': indices, 'scale': e.scale})
        else:
            p = params[e.parameter]
            locations = {(r, c) for r, c, _ in e.cells}
            candidates = [(r, c) for r in range(p.shape[0]) for c in range(p.shape[1])
                          if (r, c) not in locations] if p.numel() <= 100_000 else None
            if candidates is None:
                proposals = set()
                while len(proposals) < len(e.cells):
                    pair = (rng.randrange(p.shape[0]), rng.randrange(p.shape[1]))
                    if pair not in locations:
                        proposals.add(pair)
                candidates = sorted(proposals)
            if len(candidates) < len(e.cells):
                raise ValueError('Insufficient alternative cells for control')
            selected = rng.sample(candidates, len(e.cells))
            rows.append({'parameter': e.parameter,
                         'cells': [[r, c, delta] for (r, c), (_, _, delta) in zip(selected, e.cells)]})
    return {'version': 'wsp-edit-v0',
            'hypothesis': f'Random-index negative control for {canonical_digest(manifest)}',
            'edits': rows}


def list_editable_weights(model, *, limit: int = 24) -> list[dict]:
    """Return limited tensor metadata, not a model checkpoint or full weight dump."""
    output = []
    for name, p in model.named_parameters():
        if name.endswith(ELIGIBLE_SUFFIXES) and p.ndim == 2 and p.dtype.is_floating_point and not p.is_meta:
            with torch.no_grad():
                # Sampling avoids expensive aggregation across giant weight matrices.
                flat = p.detach().flatten()
                sample = flat[::max(1, flat.numel() // 4096)][:4096].float()
                output.append({'parameter': name, 'shape': list(p.shape),
                               'dtype': str(p.dtype), 'sample_mean': round(float(sample.mean().item()), 7),
                               'sample_std': round(float(sample.std(unbiased=False).item()), 7)})
            if len(output) >= limit:
                break
    return output
