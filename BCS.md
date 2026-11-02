# Merlino — Behavioral Concept Specification (workshop subset)

**Purpose:** demonstrate how an RDF domain model becomes an agentic system when the ability to act is bound to a governed ontology, an explicit state machine, independently validated actions, a human authority and reproducible decision evidence. This is a teaching subset, **not a certified private AOOE Rulebook package**.

## Agent and scope
- Identity: Merlino, a helpful running coach for a *fictional* athlete Alex.
- Main task: read historical evidence; propose a week of training; explain the gate result; initiate handover.
- Model: Gemma 4 E2B (optional E4B). LLM output is *untrusted candidate text/tool calls*.
- Out of scope: diagnosis, unsupervised medical decisions, real-world calendar writes, authenticated coaching, real patient or employee data.

## Ontological contract
- Legend: schema.org and SKOS terms in Bea's map, `mr:` workout concepts and typed runs. Official public `aooe:` vocabulary at `https://w3id.org/aooe#` is distinct from Bea's teaching-only legacy `https://example.org/aooe#` runtime terms.
- Map: `schema:Person`, `schema:ExerciseAction`, `schema:ExercisePlan`, stable workout type IRIs; controlled `mr:isHard` classifications.
- Rulebook: `mr:` actions/states from Bea's railroad, `mg:` candidate envelope, `mg:coachingPolicy` in a versioned policy graph, SHACL 1.0/SPARQL shapes, fail-closed external subprocess, explicit simulated human approval.

## Actions and authority
| Action | Initiator | Allowed if |
|---|---|---|
| ReadRunHistory | Gemma tool | Railroad allows initial step and gate succeeds |
| ProposePlan | Gemma tool | Evidence fresh; complete sessions; total within teaching policy; right state; gate conforms |
| ExplainLastDecision | Gemma tool | Read-only, returns gate output and current version |
| HandToHuman | Gemma tool | Railroad allows, gate conforms; stops progression |
| ApprovePlan | Separate runner action only | Explicit human-confirmed demo input; revalidated candidate; plan hash and versions bound |
| StartWeek | Separate trusted engine method only | Validated plan, same approval digest, same ontology and policy revisions, correct state |
| RevisePolicy/ReviseMeaning | Steward-only notebook methods | Explicit steward confirmation; versions change; previous approval invalidated |

## Constraints
- G1/G2/G3: candidate structure, legal railroad transition, authorized role.
- B1: stale or missing baseline prevents proposing/starting.
- B2: weekly progression ≤ configured ratio (fictional 1.10 default); independently rechecked before `StartWeek`.
- B3: reported pain prevents all continued automated actions except handover; model cannot suppress known pain in candidate graph.
- B4: approval binds exact plan content digest, policy version and ontology revision.
- B5: consecutive hard days are **warnings** based on the owned SKOS concept classification; facilitator changes the meaning to show warning changes.
- B6: plan exactly 7 distinct days, nonnegative distances, valid workout types, declared total equals sum.
- Fail closed: validator import, timeout, error or malformed report means **no commit**.

## Pre/post conditions
1. Model sends a typed tool call. Dispatcher validates method/argument allowlist.
2. Engine creates a candidate graph from trusted world state **without mutating** plan, approval, policy or ontology.
3. Gate process reads SHACL shapes and validates candidate + domain/railroad/policy RDF.
4. Independent local preflight invariants supplement but do not replace SHACL.
5. Only a conforming, authorized candidate causes state transition; denied candidate remains inspectable.
6. Decision is logged as a PROV-O-aligned activity with rule messages and version identifiers.

## Privacy & sovereignty
- Agent sees only a deliberately scoped synthetic `context()` packet and tool results.
- There is no default external inference API. Colab is still cloud-hosted; local machine recommended for genuine personal data.
- WordLift Resolve (if enabled) receives only two synthetic workout terms and synthetic sentence; no athlete profile or graph.
- Enterprise deployment must add real authenticated approval, protected validator IAM, signed policy versions, non-repudiable logs and formal evaluation.

## Negative cases
- 42.5 km vs 33 km (teaching 10% threshold): blocked before plan exists.
- 38 km against 33 km under a forged second approval: progression still blocked.
- A changed plan after approval: digest mismatch blocked.
- Missing runner confirmation: approval refused.
- Ontology or policy changed after approval: approval invalidated.
- Pain reports: future coaching actions denied; human handover remains possible.
- Unknown tool called by Gemma: rejected, never dynamically dispatched.
- Validator missing: all model-driven writes refused.
