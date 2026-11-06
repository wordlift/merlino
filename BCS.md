# Merlino — Behavioral Concept Specification (workshop subset)

**Purpose:** demonstrate how an RDF domain model becomes an agentic system when the ability to act is bound to a governed ontology, an explicit state machine, independently validated actions, a human authority and reproducible decision evidence. This is a teaching subset, **not a certified private AOOE Rulebook package**.

## Agent and scope
- Identity: Merlino, a helpful running coach for a *fictional* athlete Alex.
- Main task: read the training context; propose a week of training; explain the gate result; request a start, check-in or review when the Railroad allows it; initiate handover.
- Runtime: the workshop Colab notebook (`notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb`) and the source bundle embedded in it. This spec describes that code.
- Model: Gemma 4 E2B (optional E4B). LLM output is *untrusted candidate text/tool calls*.
- Out of scope: diagnosis, unsupervised medical decisions, real-world calendar writes, authenticated coaching, real patient or employee data.

## Ontological contract
- Legend: schema.org and SKOS terms in Bea's map, `mr:` workout concepts and typed runs. Official public `aooe:` vocabulary at `https://w3id.org/aooe#` is distinct from Bea's teaching-only legacy `https://example.org/aooe#` runtime terms.
- Map: `schema:Person` (runner), `schema:SportsEvent` (goal race), `schema:ExerciseAction` (runs, with numeric `mr:distanceKm`), stable SKOS workout type IRIs with controlled `mr:isHard` classifications. A proposed week is not yet stored as a `schema:ExercisePlan`: it exists as a candidate with seven `mg:session` nodes.
- Rulebook: `mr:` actions/states from Bea's railroad, `mg:` candidate envelope, `mg:coachingPolicy` in a versioned policy graph, SHACL 1.0/SPARQL shapes, fail-closed external subprocess, explicit simulated human approval.

## Actions and authority
Gemma sees only the tools below, and only those whose Railroad action has a transition from the current state (`tool_schemas_for_state`). A tool call is a *request*: every action is revalidated by the independent gate before anything commits.

| Action | Initiator | Allowed if |
|---|---|---|
| ReadRunHistory | Host/notebook (`read_history()`); not a Gemma tool | Railroad allows it from `Start`; gate conforms |
| Read training context | Gemma tool `get_training_context` (read-only) | Always; returns current synthetic facts and policy, never earlier episodes |
| ProposePlan | Gemma tool `propose_weekly_plan` | Right state; evidence fresh (B1); total within teaching policy (B2); seven complete sessions (B6); gate conforms |
| StartWeek | Gemma tool `start_week` (request) or trusted engine method | Right state (G2); `PlanApproval` evidence (G4); a validated proposal (B7); approval bound to the same plan digest, policy version and meaning revision (B4); B1/B2/B6 rechecked; no pain (B3) |
| AskCheckIn | Gemma tool `ask_check_in` | Week in progress; no pain hold (B3); gate conforms |
| ReviewWeek | Gemma tool `review_week` | `CheckIn` evidence from the runner (G4); no pain (B3); gate conforms |
| HandToHuman | Gemma tool `request_human_handover` | Any non-terminal state after `Start`; never blocked by pain |
| ExplainLastDecision | Gemma tool `explain_last_decision` (read-only) | Always; returns the last gate verdict and current policy/meaning revision. Not a Railroad action |
| Recall experience | Gemma tool `recall_governed_experience` (read-only) | Only when the session is created with `memory_access=True` (Experiment 8); otherwise not advertised and a forged call is denied. Past decisions are evidence, never permission |
| ApprovePlan | Runner only: `approve_as_runner(human_confirmed=True)` | Explicit human confirmation; revalidated candidate; plan digest and versions bound. No Gemma tool |
| LogRun, AnswerCheckIn, CloseWeek | Runner only: `*_as_runner(human_confirmed=True)` | Explicit human confirmation; Railroad allows. No Gemma tool |
| Revise policy / meaning | Steward-only notebook methods `revise_policy`, `revise_tempo_meaning` | Explicit steward confirmation; policy version or meaning revision changes; previous approval invalidated |
| Save / load memory | Facilitator-only `save_memory`, `load_memory(facilitator_confirmed=True)` | Local synthetic TriG; loading restores no approval, checked proposal, tool permission or state |

## Constraints
- G1/G2/G3: candidate structure, legal railroad transition, authorized role.
- G4: evidence required by the Railroad must exist in the graph, created by committed, authorized actions; the model cannot attach or select it.
- B1: stale or missing baseline prevents proposing/starting.
- B2: weekly progression ≤ configured ratio (fictional 1.10 default); independently rechecked before `StartWeek`.
- B3: reported pain prevents all continued automated actions except handover; model cannot suppress known pain in candidate graph.
- B4: approval binds exact plan content digest, policy version and ontology revision.
- B5: consecutive hard days are **warnings** based on the owned SKOS concept classification; facilitator changes the meaning to show warning changes.
- B6: plan exactly 7 distinct days, nonnegative distances, valid workout types, declared total equals sum.
- B7: only a proposal that already passed the gate may be approved or started; the host computes this, never the model.
- Fail closed: validator import, timeout, error or malformed report means **no commit**.

## Pre/post conditions
1. Model sends a typed tool call. Dispatcher validates method/argument allowlist.
2. Engine creates a candidate graph from trusted world state **without mutating** plan, approval, policy or ontology.
3. Gate process reads SHACL shapes and validates candidate + domain/railroad/policy RDF.
4. Independent local preflight invariants supplement but do not replace SHACL.
5. Only a conforming, authorized candidate causes state transition; denied candidate remains inspectable.
6. Decision is logged as a PROV-O-aligned activity with rule messages and version identifiers.

## Privacy & sovereignty
- Agent sees only a deliberately scoped synthetic `context()` packet and tool results. Prior episodes are excluded from `context()`; they reach the model only through the opt-in recall tool.
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
- `start_week` requested by Gemma without a runner approval, or with an approval for a different plan, policy or meaning: refused (G2/G4/B4/B7).
- Recall tool called when memory access is disabled: denied.
- Memory loaded from a previous session: earlier decisions are recalled, but the session starts at `Start` with no approval.
