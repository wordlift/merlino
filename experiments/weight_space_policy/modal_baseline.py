"""Run the read-only Gemma baseline on one Modal GPU after local cost preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from time import perf_counter

from .cost_gate import CostEstimate, approve_preflight

REVISION = '4adf15748de9e1a0097b6274cee9c15836c41c37'
ROOT = Path(__file__).resolve().parents[2]
GATE_REFERENCE = 'reference/bea/teaching_aooe.ttl'
GPU = 'L40S'
GPU_USD_PER_SECOND = 0.000542
CPU_CORES = 4
MEMORY_MIB = 24 * 1024
STARTUP_TIMEOUT = 300
EXECUTION_TIMEOUT = 900
ESTIMATED_BILLABLE_SECONDS = STARTUP_TIMEOUT + EXECUTION_TIMEOUT + 300
GATE_LIMIT_USD = 10.0


def collect_baseline(code_sha: str) -> dict:
    import torch
    import transformers

    from experiments.weight_space_policy.inspect import weight_window
    from experiments.weight_space_policy.patch import list_editable_weights
    from experiments.weight_space_policy.run import (
        CONDITIONS, MODEL_ID, evaluate_condition, load_gemma, summarize,
    )

    start = perf_counter()
    model, processor = load_gemma(MODEL_ID, revision=REVISION, device_map={'': 0})
    tensors = list_editable_weights(model, limit=128)
    projections = [entry for entry in tensors if entry['parameter'].endswith('mlp.down_proj.weight')]
    if not projections:
        raise RuntimeError('The pinned Gemma checkpoint has no editable MLP down projections')

    sample_indices = sorted({0, len(projections) // 2, len(projections) - 1})
    windows = [
        weight_window(model, parameter=projections[index]['parameter'], row=0, col=0)
        for index in sample_indices
    ]
    rows = []
    for condition, variant, policy in CONDITIONS:
        if variant == 'base':
            rows.extend(evaluate_condition(model, processor, condition, policy, {'changed_elements': 0}))
            print(f'{condition} finished', flush=True)

    return {
        'experiment': 'gemma-wsp-policy-v0-baseline',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'code_sha': code_sha,
        'model': MODEL_ID,
        'revision': REVISION,
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(0),
        'peak_reserved_gpu_bytes': torch.cuda.max_memory_reserved(0),
        'gpu_function_seconds': round(perf_counter() - start, 2),
        'tensor_metadata': tensors,
        'weight_windows': windows,
        'summary': summarize(rows),
        'records': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Explicitly launch one billable baseline job')
    parser.add_argument('--remaining-usd', type=float, help='Remaining authorized baseline budget, at most $10')
    parser.add_argument('--output', type=Path, default=Path('artifacts/gemma_modal_baseline.json'))
    args = parser.parse_args(argv)

    estimate = CostEstimate(
        gpu_usd_per_second=GPU_USD_PER_SECOND,
        cpu_cores=CPU_CORES,
        memory_gib=MEMORY_MIB / 1024,
        seconds=ESTIMATED_BILLABLE_SECONDS,
    )
    print(f'Estimated compute including startup and margin: ${estimate.total():.2f}', flush=True)
    if not args.launch:
        return
    if args.remaining_usd is None or not 0 <= args.remaining_usd <= GATE_LIMIT_USD:
        parser.error('--launch requires --remaining-usd between 0 and 10')
    try:
        approve_preflight(estimate, remaining_usd=args.remaining_usd, explicitly_authorized=args.launch)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error('Output already exists. Use a new path to preserve the previous run.')
    status = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=all', '--', 'experiments/weight_space_policy',
         'merlino', 'ontology', 'shapes', GATE_REFERENCE], cwd=ROOT, text=True
    )
    if status:
        parser.error('Commit changes to the experiment and validator inputs before launching')
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    import modal

    image = (
        modal.Image.debian_slim(python_version='3.12')
        .pip_install(
            'torch==2.14.1+cu130', 'torchvision==0.29.1+cu130',
            extra_index_url='https://download.pytorch.org/whl/cu130',
        )
        .pip_install('transformers==5.24.0', 'accelerate==1.14.0', 'rdflib==7.6.0', 'pyshacl==0.40.2')
        .run_commands('python -c "from transformers import AutoProcessor, AutoModelForMultimodalLM"')
        .workdir('/root/merlino-research')
    )
    for directory in ('experiments/weight_space_policy', 'merlino', 'ontology', 'shapes'):
        image = image.add_local_dir(
            ROOT / directory, f'/root/merlino-research/{directory}',
            ignore=['**/__pycache__/**', '**/.pytest_cache/**'],
        )

    image = image.add_local_file(ROOT / GATE_REFERENCE, f'/root/merlino-research/{GATE_REFERENCE}')

    app = modal.App('merlino-gemma-weight-space-baseline')
    baseline = app.function(
        image=image, gpu=GPU, cpu=(CPU_CORES, CPU_CORES), memory=(MEMORY_MIB, MEMORY_MIB),
        startup_timeout=STARTUP_TIMEOUT, timeout=EXECUTION_TIMEOUT, retries=0,
        min_containers=0, max_containers=1, scaledown_window=2, serialized=True,
    )(collect_baseline)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with modal.enable_output(), app.run():
        report = baseline.remote(code_sha)
    report['estimated_compute_usd'] = estimate.total()
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    print(f'Baseline report written to {args.output}')


if __name__ == '__main__':
    main()
