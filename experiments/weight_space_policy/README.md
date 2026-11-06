# Weight-Space Policy Programming (WSP) — independent Gemma experiment

**Research-only · not part of Merlino's workshop · no merge into the governed agent without separate review.**

## Question

Can an LLM **write a new organizational policy into a Gemma model's weights**, without gradients, so its **first proposed action** respects a changed limit even when the prompt contains no current limit (or an outdated one)?

This is **not** an effort to replace AOOE governance. Policy *authority* remains in the RDF/SHACL graph. An edited language model can still be incorrect, and a passing gate validates a candidate, not the model's internal rule.

## Why this is a genuine open question

Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) (2 Nov 2026), report that a frontier LLM directly modified small portions of a Qwen3.5-0.8B checkpoint without gradient descent, primarily by eliminating repetitive reasoning loops. **That is not a demonstration of precise arbitrary policy updates.** The same report found no established mathematical improvement through its direct edits to Gemma 3 270M. We are testing a different task and **Gemma 4 E2B**; results must not be inferred from their benchmarks.

## Status

- [x] Independent branch from private `main` (no use of PR #3's episodic-memory changes)
- [x] A reversible, bounded **real tensor editing** tool supports specific raw cell deltas and MLP projection axis scaling
- [x] Controlled current-policy / stale-policy / no-policy prompts; no gradients, LoRA or optimizer
- [x] Synthetic 33-km runner, v1 = 10% increase, v2 = 5%; fresh, independent **PySHACL validator** for every proposed action
- [x] Identical-base and random-index-edit control with exact weight restoration
- [x] CPU unit tests for patch contract, rollback, malformed values and controls
- [ ] A human or dedicated editing LLM examines **actual Gemma tensor values**, states a mechanistic hypothesis and supplies a checked numeric patch JSON
- [ ] GPU Gemma 4 run, pinned checkpoint, multi-candidate comparisons
- [ ] Independent held-out examples and capability-regression checks before any positive research claim

**There is deliberately no claimed successful Gemma weight patch in this repository.** The patch is a proposed intervention, not a demonstrated capability.

## Run from the Merlino repository root

CPU preparation:

```bash
python -m pip install pytest torch rdflib pyshacl requests
python -m pytest -q experiments/weight_space_policy/tests
python -m experiments.weight_space_policy.run --patch my_candidate.json --dry-run
```

Inspect **actual model parameters** on a CUDA GPU (strongly prefer A100 40 GB rather than T4 for BF16 edits):

```bash
python -m pip install transformers accelerate safetensors
python -m experiments.weight_space_policy.run --inspect --model google/gemma-4-E2B-it --revision <PINNED_CHECKPOINT_SHA>
```

Then author a patch manifest after examining weight shapes and a bounded raw-weight window. For a numeric patch use **actual** parameter names from the above output; never copy guessed names from the sample below.

```json
{
  "version": "wsp-edit-v0",
  "hypothesis": "A particular observed MLP writer may contribute to reliance on the previous 10 percent threshold.",
  "edits": [
    {"parameter": "<INSPECTED_NAME>.mlp.down_proj.weight", "axis": 1, "indices": [1], "scale": 0.98}
  ]
}
```

The above JSON is a **schema illustration only**, not a valid, evaluated Gemma patch. Alternative direct numerical edits can use `{ "parameter": "...", "cells": [[row, col, delta], ...] }` with at most 4,096 cells per tensor.

Run the 30 prompt / 6-arm smoke trial (5 probes × 6 conditions):

```bash
python -m experiments.weight_space_policy.run \
  --patch my_candidate.json \
  --model google/gemma-4-E2B-it \
  --revision <PINNED_CHECKPOINT_SHA> \
  --output artifacts/wsp-policy-smoke.json
```

The evaluation saves a JSON record of **all raw outputs, parsed proposals, B2 verdicts, patch digest, model revision, policy, software versions and hardware**. No model checkpoint is saved or published. Distinct conditions run sequentially on the **same base weights**, restored byte-for-byte after each patch arm.

## The six arms (all governed by the SAME v2 SHACL gate)

| Arm | Model | Prompt policy | Purpose |
| --- | --- | --- | --- |
| Base/no | Base Gemma | None | What it does without rule knowledge |
| Base/old | Base Gemma | Stale 10% | Vulnerability to stale instruction |
| Base/new | Base Gemma | Current 5% | Prompt-only, oracle-aligned comparison |
| Edited/no | Directly patched Gemma | None | Has the new policy entered weight behavior? |
| Edited/old | Directly patched Gemma | Stale 10% | Can the edit resist an outdated policy? |
| Random/no | Random-coordinate edit of same tensor/count/scale | None | Is improvement specific to the proposed intervention? |

The random-control patch matches **tensor, axis, number of coordinates and multiplicative factor**, not numerical delta norm. A norm-matched control and repeated interventions are required before mechanistic attribution.

## Interpretation

- **Primary pilot measurement:** accepted **first proposals** (not just refusals) under the v2 policy and valid JSON schema. 34.6 km is eligible for the fictional 33×1.05 policy; 34.7 is not.
- **Important negative outcome:** model cannot encode or reliably retrieve a new numeric limit from limited weight editing. That would be informative and does not weaken AOOE.
- **Confound:** a lower-number bias can pass the gate without representing 5% policy. Test unseen baselines, alternative policy changes and unrelated tasks before claims of internalization.
- **Under no circumstances:** model weight edits authenticate people, change SHACL, grant approval, or commit real-world actions.
- **No patient, personal training or production data.** Gemma runs inside a cloud-hosted Colab runtime if the optional notebook is used.

## Relationship to existing WordLift work

- **Merlino/AOOE:** the graph still owns and independently validates policy and action authority.
- **Hippocampus:** tests the complement between explicit, persistent verified evidence and parametric behavior. This is **not** a Hippocampus training result or verified online learning.
- **RLM-on-KG:** evidence navigation could eventually help select weight-editing targets, but the present protocol does **not** implement RLM-on-KG.

This branch deliberately does **not** change `merlino/`, `ontology/`, `shapes/`, the existing notebook, or the unmerged episodic-memory PR.
