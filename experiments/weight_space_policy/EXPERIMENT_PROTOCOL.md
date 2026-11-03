# WSP-0: pre-run experiment plan

**State:** scaffold and CPU tests only. Write this plan and freeze a revision *before* inspecting held-out data or announcing a result.

## Target and counterfactual

Question: Can a **directly edited** Gemma 4 E2B predict a valid next-week proposal under **new** graph policy v2 (1.05× 33 km = 34.65 km) when policy text is absent or old (v1 = 1.10×), more reliably than the original model and a random-coordinate edit?

The *independent* official policy lives in `ontology/policy.ttl` + the Merlino SHACL execution path; the harness uses `revise_policy(..., steward_confirmed=True)` **only** on each fresh *synthetic* Merlino evaluator. The model cannot edit the policy, gate or approvals.

## Proposed mechanism

A frontier weight-programmer (or human researcher) reads **bounded, actual** tensor measurements and mistakes, then proposes a JSON manifest of specific numerical changes to MLP parameters. The patcher writes those numbers into a **full-precision** Gemma checkpoint directly. **No gradient, loss minimization, LoRA, optimizer or hidden finetune** is run. This probes whether existing features can be redirected to express a new policy; it does not assume such a feature exists.

Allowed operations are (1) scaling at most 64 rows/columns in selected MLP matrices, or (2) adding at most 4,096 explicit values to chosen cells. At most 4 tensors, 262,144 modified scalar weights, and at most 0.05% of the model parameters. Each candidate is reversible and logged. These limits prevent uncontrolled artifacts; they are not proofs of harmlessness.

## Pilot data (NOT a benchmark)

Five fictional text prompts requesting 42.5 / 34.7 / 36.0 / 34.0 / 40.0 km after 33 km. Prompt wording includes both neutral and stale-rule pressure. The six arms have the same weights except for the temporary edit, and the exact same v2 PySHACL oracle. Deterministic generation settings (`do_sample=False`, max 160 new tokens) are fixed across conditions.

Primary outputs: (a) strict JSON parse validity, (b) first candidate SHACL acceptance, (c) rate of B2 blocks, (d) proposal distribution and overconservatism, (e) raw responses and latency. Secondary: edits' location, count, changes vs random-index control, versions/hashes.

## Gate to confirmatory evaluation

Do not claim the policy was encoded in weights merely because a few first proposals pass SHACL. A confirmatory run requires **at least 30 additional held-out prompts with other synthetic baseline magnitudes, unseen requests and paraphrases**, different policy ratios, unrelated-task performance checks, multiple patch attempts with precommitted selection criteria, and norm-matched/random-edit controls. This pilot currently uses a fixed 33 km history; the additional baselines require graph-derived fixture changes (never `engine.baseline_km` overwritten manually).

If the editor is given feedback on pilot examples, those examples are **development data**. Freeze both patch and new evaluation examples before testing generalization. Report *failures, regressions and non-responses*, not just validity after the gate. No paraphrasing of SHACL scores as evidence of safety.

## Criteria and limitations

A viable finding would require a practically large, replicated improvement in *proposals accepted on the first attempt* relative to **both** base/no-rule and random/no-rule arms, while not degrading unrelated abilities and while adapting to changed, unseen policy values. Prompt-only current-policy remains an important upper/reference arm; weight edits must show a benefit beyond learning a single memorized number.

Keep the external SHACL gate **even if editing succeeds**. No direct weight patch can implement authenticated human consent, version control, provenance or policy authority. A cautious null is an expected outcome.

## Source scope

- Weight-space programming blog (2 Nov 2026) demonstrates weight intervention on **Qwen3.5-0.8B**; its reported improvement primarily solves reasoning loops. It does **not** verify numeric policy updates or behavioral changes on Gemma 4.
- It reports no established improvement from direct edits to Gemma 3 270M on MATH-500. This is further reason not to predict a positive result here.
- Gemma 4 E2B BF16 inference requires roughly **11.4 GB** of GPU memory by Google's approximate table; extra overhead for PyTorch, patch inspection and generation makes an A100 preferable to a 16 GB T4. Standard bitsandbytes 4-bit modules are not safely writable through this patcher and are explicitly unsupported.
