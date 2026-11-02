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
Uses Google's Gemma 4 native function declaration schema via `AutoProcessor.apply_chat_template(tools=...)`, generation with `AutoModelForMultimodalLM`, and `processor.parse_response` if present. Only tool names in an explicitly declared registry are executed; the tool selection is untrusted even if it arrives in a structured model frame. The model never sees API credentials or graph infrastructure management endpoints.

The model integration is **not yet GPU-tested**. Try E2B first (T4 + 4-bit), record compatibility, then benchmark E4B on available GPU memory. If quantization has compatibility issues in an installed Transformers version, consult the official Gemma function-calling notebook and use its documented loader with a sufficiently large GPU rather than silently using a remote model API.

## Licenses and publication
Bea's workshop code is user-supplied for synthesis; obtain contributor consent/license before publicly distributing a derivative. AOOE public vocabulary is CC BY 4.0 but the private methodology is not. WoGraph OSS's pom declares AGPL-3.0 and its repo was private when inspected. No third-party model weights are bundled; users must comply with Gemma model license.
