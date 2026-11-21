# Comparative weight-space programming: Gemma 4 E2B vs Inkling 2 Mini

Status: implementation plan only; **no Modal job, model download, or paid API invocation authorized by this file**. Research remains isolated in PR #4 and must not alter `merlino/`, `ontology/`, `shapes/`, or production notebooks.

## Existing baseline (verified in PR #4)
- `run.py`: Gemma-only `AutoModelForMultimodalLM` loader, five synthetic probes, six evaluation arms, independent PySHACL gate.
- `patch.py`: reversible BF16/FP tensor edits; only `.mlp.{down,up,gate}_proj.weight` suffixes; randomized control not norm-matched.
- `inspect.py`: bounded 32x32 tensor windows.
- No successful GPU weight patch established. Do not label generated JSON patches as evidence.

## Gate 0 — zero-cost feasibility (required before any Modal launch)
1. Verify the official exact Inkling 2 Mini model ID, checkpoint SHA, license, parameter count, architecture, tensor layout, tokenizer, inference compatibility, and whether its weights are actually downloadable. Do not infer from model branding.
2. Verify Gemma 4 E2B checkpoint revision and required gated access.
3. Pin compatible `torch`, `transformers`, and architecture-specific runtime versions; avoid arbitrary `trust_remote_code`.
4. Fetch current Modal GPU/CPU/RAM/volume prices; compute a worst-case hourly rate from all billed resources, not just GPU. Explicitly verify that the chosen GPU has enough VRAM for unquantized editable weights and inference headroom.
5. Record exact pricing timestamp, source URLs, checkpoint SHAs and model licenses in an immutable run manifest. If any item fails, stop without spending.

## Implementation sequence
A. Refactor model loading behind an adapter (`load`, `generate`, `list_editable_tensors`, `inspect_window`) with explicit architecture-specific code; keep Gemma behavior unchanged. No assumption that Inkling uses dense MLP suffixes.
B. Extend the patch validator only after examining real Inkling tensors. Explicit allowlist of 2-D floating-point tensors; reject quantized, shared/aliased, routed metadata and unknown layouts. Restore bytes exactly after each arm; serialize experiments (no concurrent inference).
C. Add an expert-routing trace for Inkling if supported, otherwise mark unavailable; track expert activation before/after edits and routing consistency.
D. Keep the same six arms and fictional 33 km policy. Add held-out baseline distances, policy reversals, paraphrases, and unrelated-task regression probes; freeze prompts and seeds before editing.
E. Replace coordinate materialization with bounded random sampling. Add random controls matched on effective changed element count and measured delta L2 (with a tolerance and explicit failures). Record BF16 rounding-induced no-ops.
F. Write `modal_compare.py` as an opt-in runner, not a default CI job. It must enforce an explicit `--allow-paid-run` and `--max-usd`, hard runtime deadline, max candidate count, concurrency=1, idle scale-to-zero, and a persistent local cost ledger. Budget is an estimated guard, **not** a hard provider billing cap. Require user approval before the first paid launch.
G. Output per-model JSON: revision, license, architecture, software versions, GPU, billed time, estimated charges, patch SHA/digest, tensor names and delta norms, raw outputs, SHACL results, held-out pass rates, regressions and limitations.

## Modal cost gate
- Phase 0: $0; repository/metadata inspection and CPU-only tests.
- Phase 1: at most $10 *estimated all-in* for one model loading and baseline smoke, only after verified hourly pricing.
- Phase 2: at most $40 *additional estimated all-in* for bounded candidate exploration and comparison, conditional on phase 1 succeeding.
- Total initial authorization ceiling proposed: $50, **not yet approved for expenditure**.
- Preflight: `ceil((startup_seconds + expected_inference_seconds + shutdown_seconds)/3600 * (gpu_hourly + cpu_hourly + ram_hourly)) + storage + API` must be below remaining budget. Stop if forecast exceeds remaining allowance; log actual billing when available.
- If Inkling does not fit on the affordable single GPU or requires expensive multi-GPU hosting, stop and report revised estimate rather than silently switching hardware.
- Keep GPU container warm only while needed, enforce timeout, no background persistent GPU, and no checkpoint publishing.

## Acceptance criteria
- CPU CI green and deterministic dry run for both adapters.
- Both models load with pinned revisions and produce parseable baseline responses.
- Identical synthetic six-arm protocol with reversible, measurable real weight edits and random controls.
- At least one candidate per model plus held-out/regression evidence; no claims of policy encoding without significant and reproducible improvement over prompt-only and random controls.
- Merlino policy authority remains exclusively in independent graph/SHACL validation.
- PR #4 remains draft and unmerged until review.

## First implementation task
Implement adapter interface and CPU fakes/tests, followed by zero-cost model metadata verification. **Do not run Modal yet.**
