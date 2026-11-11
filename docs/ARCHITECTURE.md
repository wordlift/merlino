# Architecture decisions and Bea integration

## Source and evolution
Bea's `Merlin Running Pal` contributed `merlin_map.ttl` and `merlin_railroad.ttl` unmodified, and `aooe.ttl` + original `gate_shapes.ttl` retained under `reference/bea/` for discussion, attribution and a before/after review. The original `merlin/tools.py` mutates plan and approval state before SHACL; **this new implementation does not execute it**. The hardened engine uses a new candidate schema and gate while retaining Bea's states, transitions, roles and workout types.

## Why a new execution path
`Gemma4Transformers → MerlinoAgent → Merlino engine → gate_worker subprocess (PySHACL) → state commit → RDF trace`.

- Proposal generation is side-effect-free and deterministic given chosen numeric amount/style. No LLM can directly call Python functions, approve, edit graph/policy or write WordLift resources.
- Runner confirmation is a separate call, not available in model tool schemas.
- Each plan is revalidated on start; approval is bound to SHA-256 over structured plan + named ontology revision and policy version.
- Same-policy or same-plan *names* are not sufficient: the digest must match exact sessions.
- The gate is fail-closed. Same-UID subprocess is useful to explain separation, not by itself an access-control boundary for regulated workloads.
- SHACL gives explicit constraints; the additional Python checks are defense in depth and can report errors when SHACL is unavailable, but never override a failed gate.

## Agent environment, memory and verification

**Semantic world:** `merlino.map` (RDF observations + meanings), `merlino.policy`
(versioned governance), and the Railroad (authorized state transitions).
**Episodic memory:** RDF `prov` decision traces, optionally exported alongside
the world to a local `.trig` file. New sessions can recall previous outcomes,
but have no old approvals or execution authority.

**Separation of responsibilities:** the gate runs SHACL and returns valid/invalid
results and codes. The `ScriptedMerlin` deterministic policy or a Gemma tool
client decides whether to propose a corrected candidate, request a human or
stop. SHACL does not calculate a repair.

**Research context, NOT workshop implementation:** WordLift's [RLM-on-KG]
(https://github.com/wordlift/rlm-on-kg) explores adaptive graph navigation;
Hippocampus investigates learning from verified navigation/traces. Neither
full research system is wired into this demo. See `docs/SEMANTIC_MEMORY.md`.

## Enterprise mapping
| Demonstration | Production boundary |
| --- | --- |
| Editable RDF in notebook | versioned ontology repo, CODEOWNERS, signed deployment |
| Separate Python validator | separate restricted gate service with independent IAM and SHACL policy releases |
| Boolean `human_confirmed` | authenticated runner identity + approval signature/attestation |
| Local JSON/PROV-O trace | append-only access-controlled audit with correlation/retention |
| Colab Gemma GPU | on-premises / private GPU, network-isolated inference |
| RDFLib | WoGraph OSS (independent Java stack) or managed WordLift, explicit data export |

## Tool-call implementation
Uses Google's Gemma 4 native function declaration schema via `AutoProcessor.apply_chat_template(tools=...)`, generation with `AutoModelForMultimodalLM`, and `processor.parse_response(generation, prefix=template, tools=tools)` when available (the prefix and tool schemas are required for Gemma 4's chat template; `tests/test_gemma_runtime_contract.py` covers it). Only tool names in an explicitly declared registry are executed; the tool selection is untrusted even if it arrives in a structured model frame. The model never sees API credentials or graph infrastructure management endpoints.

The model integration has been exercised in Colab during development, but a full fresh-GPU rehearsal is still a release blocker (`docs/RELEASE_APPROVALS.md`). Try E2B first (T4 + 4-bit), record compatibility, then benchmark E4B on available GPU memory. If quantization has compatibility issues in an installed Transformers version, consult the official Gemma function-calling notebook and use its documented loader with a sufficiently large GPU rather than silently using a remote model API.

## Licenses and publication
Bea's workshop code is user-supplied for synthesis; obtain contributor consent/license before publicly distributing a derivative. AOOE public vocabulary is CC BY 4.0 but the private methodology is not. WoGraph OSS's pom declares AGPL-3.0 and its repo was private when inspected. No third-party model weights are bundled; users must comply with Gemma model license.

## CDL 2026 full life cycle and reproducibility

The engine now implements all ten actions in Bea's nine-state Railroad. Merlin
(agent role) can read history, propose, start (after approval), ask a check-in, review and
escalate; reading history is a host step, and the rest reach Gemma as tools only when the
Railroad allows them (see `BCS.md`). Runner-only actions log runs, answer check-ins, approve exact plans
and close weeks. The `merlino.scenarios` module replays seven paired cases.
The `NoRulesMerlino` class is **only** an isolated synthetic shadow and must
never be exposed as an LLM tool, deployment option or network service.

The `merlino.history` module derives training baseline and data age from the
Map's RDF observations. `merlino.engine` binds the latest runner check-in and
active pain hold into the authoritative candidate context; the LLM has no
path to supply a replacement. The independent SHACL gate evaluates four
generic rules G1–G4 and seven coaching/integrity rules B1–B7. B2 computes
actual volume from sessions, not a declared total. Versioned approvals bind
to the plan digest, policy and ontology meaning revision.

PROV-O-aligned Observation/Decision/Validation/Execution nodes record the
outcome for each attempted step, and the dedicated notebook runs SPARQL
queries to explain the S5 failure. This is an editable local teaching trace,
not a production tamper-evident log.
