# Gemma 4 E2B weight-space programming — Modal execution plan

**Research only.** Scope: `google/gemma-4-E2B-it` only. PR #4 remains draft/unmerged; no changes to production `merlino/`, `ontology/`, or `shapes/`.

## Current verified code
- `run.py` has six policy-evaluation arms, five fictional requests and an independent PySHACL gate.
- `patch.py` applies reversible bounded edits to floating-point MLP tensors; random-coordinate control is not norm-matched.
- `inspect.py` provides bounded inspection windows.
- `cost_gate.py` is an **offline estimate** with explicit authorization checks, not a billing enforcement service.
- There is **no verified successful GPU weight patch**.

## Gate A: CPU, $0
1. Run `python -m pytest -q experiments/weight_space_policy/tests` with pinned dependencies; fix failures.
2. The public Gemma checkpoint is pinned to `4adf15748de9e1a0097b6274cee9c15836c41c37`. Its safetensors header lists 105 BF16 MLP projections, with down projections shaped `[1536, 6144]`. The checkpoint is 10.25 GB. The Modal run records tensor metadata and bounded samples from the loaded model.
3. Before proposing trial patches, add held-out baseline distances, policy reversals, paraphrases and unrelated-task regression tests; freeze evaluation dataset before generating candidate patches.
4. Add random controls matched on observed edit L2 and effective changed-element count, not only coordinate count/scale.
5. Add an explicit patch-manifest dry run and byte-exact restoration test.
6. Record pinned dependencies, software versions and code SHA.

## Gate B: Modal preflight, estimated <= $10
1. Verify **live** Modal GPU, CPU, RAM and storage prices and access to checkpoint; calculate all-in cost.
2. Confirm unquantized model fits on selected single GPU with KV cache and editing headroom. If not, stop and revise plan; do not silently upgrade.
3. The opt-in Modal runner uses one L40S, a 15-minute execution timeout, a five-minute startup timeout, and no application retries or warm containers. No default CI GPU launch.
4. Launch only with explicit authorization and an estimated preflight below $10. Provider charges can differ; budget check is not a hard billing cap.
5. Inspect real parameters and record baseline responses; do not invent numeric weight patch values.

### Run the baseline

From the root of a clean checkout using an authenticated Modal CLI profile:

```sh
python -m pip install modal==1.6.4
python -m experiments.weight_space_policy.modal_baseline
python -m experiments.weight_space_policy.modal_baseline --launch --remaining-usd 10
```

The offline estimate is $0.97 for one L40S, four CPU cores, and 24 GiB RAM over 25 minutes, including startup and five minutes of margin, using the listed [Modal rates](https://modal.com/pricing). Confirm current rates before launch. Image construction, provider-level rescheduling, and actual billing can differ from this estimate. The baseline writes `artifacts/gemma_modal_baseline.json` with all 15 unedited-model responses, PySHACL decisions, tensor samples, and GPU timings. The output path can be changed with `--output` for a subsequent run.

## Gate C: candidate trial, additional estimated <= $40
1. Human/editor LLM documents a mechanistic hypothesis based on inspected tensor values, then proposes bounded numeric JSON.
2. Run six arms on identical checkpoint and prompts, restoring weights between arms.
3. Repeat with multiple candidates/seeds and matched random controls; compare held-out gate pass rate, refusal rate, stale-policy resistance and unrelated-task regressions.
4. Publish raw outputs, SHACL verdicts, delta norms, checkpoint SHA and actual measured charges. No positive claim without reproducible evidence.

**Proposed initial ceiling: $50 total, not a guarantee or preauthorization for actual spending.** No paid job is launched by this document.
