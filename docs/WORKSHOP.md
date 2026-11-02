# AOOE in Practice: Semantic Sovereignty with Merlino

**Question:** How does ontology engineering evolve when an agent can make consequential decisions in a regulated enterprise?

| Time | Exercise | Deliverable |
| --- | --- | --- |
| 0–15 min | Private data and semantic sovereignty | Model/data boundary |
| 15–35 | Legend + Map: competency questions | Reproducible RDF queries |
| 35–55 | Real Gemma tool proposal or deterministic fallback | Candidate action |
| 55–75 | 42.5 km progression blocked by independent SHACL | Violation + trace |
| 75–95 | Change `TempoRun` hard-session classification | Before/after semantic result |
| 95–110 | Attempt to bypass approval; inspect decision provenance | Audit / authorization boundary |
| 110–120 | WoGraph OSS, WordLift Resolve and enterprise deployment | Control and data-flow choices |

**Preferred workshop:** Gemma E2B on a Colab GPU with a deterministic CPU fallback. The fallback still runs the real SHACL gate; it is not falsely presented as LLM inference. Perform a full GPU rehearsal before the event.

**Main message:** Gemma proposes; RDF defines; SHACL validates; humans authorize.

## Discussion prompts

- What do companies own when they own the model, but not its meanings?
- Who is allowed to publish an ontology change that modifies an agent's behavior?
- Why is out-of-band validation different from asking the model to check itself?
- Which data may leave the enterprise boundary, and to whom?

## Enterprise mapping

This notebook runs on a hosted Colab VM, not a private computer. SHACL runs in a subprocess with the same local OS permissions, not an enterprise IAM boundary. Approval is simulated. In production, separate and authorize the policy service, sign ontology releases, use an immutable audit ledger, and deploy inference/data within the organization's chosen infrastructure.

## Updated 120-minute masterclass flow

Use Bea's original title **From Map to Railroad**. The updated exercises are
sequenced as Map (20 min), Railroad and competency checks (15 min), independent
SHACL rules (20 min), paired S1–S7 scenarios (25 min), S5 and S4 forensic
replays (15 min), optional Gemma agent (15 min), and Semantic Sovereignty
(10 min). See [`BEA_CASE_COVERAGE.md`](BEA_CASE_COVERAGE.md) for measured
scenario expectations and limitations.

Keep the scripted replay as the CPU-friendly default and the Gemma 4 GPU
runtime as an opt-in interactive demonstration. Never call scripted results
LLM benchmark results. Do not request real athlete data on hosted Colab.
