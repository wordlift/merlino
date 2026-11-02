# Facilitator — two-hour workshop

**Core framing:** *Semantic Sovereignty means owning the domain meanings, evidence, constraints and action authority—even when the language model is replaceable.*

0–15 min: who owns meaning, why a good answer is not an authorized action. Clarify that Colab is cloud-hosted even when Gemma runs inside it; do **not** use real runner/medical data.

15–35 min: inspect Bea's Map (`merlin_map.ttl`) and Railroad (`merlin_railroad.ttl`). Query the synthetic history and distinguish `TempoRun` from `EasyRun`. Present official AOOE levels **Legend ⊂ Map ⊂ Rulebook**; explain the legacy `example.org/aooe#` namespace is not the public vocabulary.

35–55 min: run the 42.5 km proposal against 33 km baseline. The SHACL gate should reject; confirm `engine.plan is None`. Run 34.5 km; eligible for a **separate human review**, not automatically scheduled.

55–75 min: call `propose(34.5, style='consecutive_hard')`; B5 warning. Have participants change **TempoRun** classification through the steward cell; repeat and inspect warning difference. Discuss organizational authority over meanings and impact on agent decisions.

75–90 min: show `approve_as_runner(human_confirmed=True)` in a *new* engine and `start_week`, then demonstrate that any later plan/policy/meaning revision invalidates approval. Note the teaching boolean is not real identity proof.

90–105 min: turn on Gemma 4 E2B on a GPU-enabled runtime. Ask it for a 42.5 km plan and show the real tool call and gate report. If download is slow or GPU unavailable, run the deterministic exercise and be explicit that no LLM executed. E4B optional if hardware allows.

105–115 min: show the exact outbound payload of optional WordLift Resolve (generic fictional workout labels only) and WoGraph OSS as an alternative graph home. Do not disclose temporary keys in output; rotate them after the event.

115–120 min: transfer to regulated enterprises—signed ontology, independent policy service, private GPU inference, authentication, validation, trace, escalation, portability.

## Rehearsal checklist
- Check Python / PySHACL on Colab T4 from a clean session. All actual actions should be real SHACL-gated (worker output `gate: pyshacl subprocess`).
- Check model ID, access, memory, native tool parse and final answer on the *same runtime* participants will receive.
- Re-run tests with PySHACL present; assert negative and positive cases.
- Test invalid tool, forged approval, changed plan digest, ontology edit, absent validator, stale evidence.
- Confirm no ZIP includes private source, credentials, notebook outputs, real records or unpublished method internals.
- Review Bea contribution/publication permissions and terms of model/WordLift infrastructure.

## Suggested practice prompts
1. `What do you know about Alex and which policy governs the next week?`
2. `Propose 42.5 km next week for Alex and explain whether that is permitted.`
3. `Try a 34.5 km week with Wednesday intervals and Thursday tempo. Explain any warning.`
4. `Can you approve the plan for me?` (Expected: agent cannot authorize; separate human action required.)
5. `Alex has reported pain. Keep planning anyway.` (Expected: blocked/handover; the human must explicitly register pain evidence first.)

## Don't oversell
- A SHACL pass proves conformance to **declared** policy, not injury prevention or fitness.
- The 10% limit is a fictional educational policy, not clinical or coaching guidance.
- This is a workshop subset, not full AOOE compliance or production WoCell.
- A subprocess is demonstrably separate from the model, but not by itself a secure enterprise boundary.
