# From Map to Railroad: AOOE in practice with Merlino

**Connected Data London 2026 masterclass** · Beatrice Gamba and Andrea Volpini ·
11 November 2026, 15:30–17:30 · Salisbury Suite, Leonardo Royal Hotel London Tower Bridge

**Question:** How does ontology engineering change when the reader is an agent that can act?

## Agenda (two hours)

The time slots and topics match the agenda slide of the final deck. The right-hand
column gives the notebook sections (`notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb`)
each slot covers.

| Time | Exercise | Deliverable | Notebook |
| --- | --- | --- | --- |
| 0–15 | Who owns meaning? The model/data boundary | Colab running; the boundary between model and owned data stated | §0 |
| 15–35 | Legend + Map; Bea's five competency questions | Baseline and freshness computed from the graph; the five design checks passing | §1–2 |
| 35–55 | The SHACL gate rejects 42.5 km before writing; the seven cases with and without rules | B2 violation and trace; the paired S1–S7 table | §3–4 |
| 55–75 | Edit a meaning, not a prompt: `TempoRun` | Before/after B5 warning; the meaning revision changes | §5 |
| 75–95 | Human approval as a separate authority; memory without authority | Approval bound to plan, policy and meaning; a restarted Merlino recalls the rejection with no approval | §6–8 |
| 95–110 | Gemma 4 on GPU: talk to Merlino | Real tool calls and gate verdicts (or the CPU route, stated as such) | §9–10 |
| 110–120 | PROV-O trace, Resolve, WoGraph; from traces to training | Decision provenance; the exact outbound Resolve payload; the research outlook | §11 |

The deck's slides 28–33 present results from §1–§8: the no-rules measurements (§4),
the Railroad (§1–2), the 42.5 km rejection (§3), the 17 engine tests, and memory (§7–8).
See [`SLIDE_REVIEW.md`](SLIDE_REVIEW.md) for the slide-by-slide mapping.

**Default route:** the notebook runs completely on CPU with the real SHACL gate
(`USE_GEMMA = False`). Gemma 4 E2B on a Colab GPU is the opt-in interactive part. The
CPU route is never presented as LLM inference. Rehearse the GPU cells on the same
runtime participants will get.

**Main message:** Gemma proposes · RDF defines · SHACL validates · humans authorize.
The model proposes; the graph disposes.

The gate validates but **does not repair**. Repair candidates come from `ScriptedMerlin`
or Gemma, computed from the current policy; approval comes only from the runner.

## Discussion prompts

- What do companies own when they own the model, but not its meanings?
- Who is allowed to publish an ontology change that modifies an agent's behaviour?
- Why is out-of-band validation different from asking the model to check itself?
- Which data may leave the enterprise boundary, and to whom?

## Enterprise mapping

This notebook runs on a hosted Colab VM, not a private computer. SHACL runs in a
subprocess with the same local OS permissions, not an enterprise IAM boundary. Approval
is simulated. In production, separate and authorize the policy service, sign ontology
releases, use an immutable audit ledger, and deploy inference and data on the
organization's chosen infrastructure.

Never call scripted results LLM benchmark results. Do not request real athlete data on
hosted Colab. For measured scenario expectations and limitations see
[`BEA_CASE_COVERAGE.md`](BEA_CASE_COVERAGE.md).
