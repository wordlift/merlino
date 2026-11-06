# Final CDL 2026 deck ↔ repository alignment

**Deck:** *From Map to Railroad — Agent-Oriented Ontology Engineering for Governed AI Systems*,
Beatrice Gamba and Andrea Volpini, Connected Data London, 11 November 2026, 15:30–17:30
(37 slides). This file maps every slide that makes a claim about Merlino to the notebook
section, file or test that backs it. Checked against `main` on 6 November 2026: 54 tests
pass and CI runs the whole notebook on CPU.

Notebook sections (`notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb`): §0 setup ·
§1 Legend + Map · §2 Railroad and five competency questions · §3 SHACL gate ·
§4 seven paired scenarios · §5 edit a meaning (TempoRun) · §6 human approval ·
§7 semantic memory · §8 Experiment 8 · §9 Gemma 4 on GPU · §10 talk to Merlino ·
§11 PROV-O trace, Resolve, WoGraph.

## Slide by slide

| # | Slide | Claim | Backed by | Status |
| --- | --- | --- | --- | --- |
| 6 | One concept. One contract. | A full BCS has eleven sections, definition to reward, with K-, M- and B-CQs and shapes with counterexamples | `BCS.md` is a workshop subset; its Know / Remember / Do questions are listed in its "Competency questions" section | ✓ subset, stated in `BCS.md` |
| 9 | Ontology engineering for a reader that acts | Know / Remember / Do; vocabulary `w3id.org/aooe#` | `BCS.md` competency questions; README links the persistent namespace | ✓ |
| 11 | Each level adds a right | Legend: addressability, workout terms and identifiers · Map: readability, athlete, sessions, plan, evidence · Rulebook: agency, the Railroad, the SHACL gate, human sign-off | README and `docs/TECHNICAL_README.md` level tables use the same wording | ✓ |
| 26 | Agenda | Seven time slots, with notebook § references | `docs/WORKSHOP.md` uses the same slots | **Deck fix:** the § numbers predate the current notebook (see below) |
| 27 | Bea's coach. Bea's data. Any model. | Re-engineered from Beatrice Gamba's Merlin Running Pal; open-weights Gemma | README credits; `NOTICE.md` | ✓ |
| 28 | Without rules, every mistake went through | S2 42.5 km started · S3 21-day-old data · S4 two weeks, the second after pain · S7 zero approvals | Bea's original notebook, step 4; reproduced in Merlino §4 no-rules shadow (`merlino/scenarios.py`, `tests/test_workshop_cases.py::test_all_seven_cases_shadow_observes_violation`) | ✓ |
| 29 | What exists is not what may happen next | 9 states, 10 actions, 2 roles, 3 evidence types; notebook §1–2 | `ontology/merlin_railroad.ttl`; §1–2 | ✓ |
| 30 | The model proposes. The graph disposes. | Propose / define / validate / authorize; 42.5 km on 33 km rejected, nothing scheduled | §3; `tests/test_engine.py::test_proposal_rejection_does_not_write_plan` | ✓ |
| 31 | When the model cheats, the gate holds | 3 blocked, 1 handed to a human, 2 fail closed; 17/17 tests in `tests/test_engine.py`; a meaning or policy change revokes approval | `tests/test_engine.py` has exactly 17 tests (the full suite has 54) | ✓ |
| 32 | It remembers the why, not just the what | Map → Railroad → Gate → Memory → Adaptation; notebook §3–§8; next proposal 36 km | §3–§8; 36 km is the S2 repair under the 1.10 policy (§4). Experiment 8 (§8) tightens the policy to 1.05 and proposes 34.6 km | ✓ (the "training data" line: see below) |
| 33 | Merlino's week, on the rails | 42.5 km ✕ B2, 36 km ✓, pain ✕ B3, exit rail from every stop after Start, start week needs approval; notebook §1–§5 | `ontology/merlin_railroad.ttl` t14–t19; `shapes/sovereign_gate.ttl` | ✓ |
| 34 | Four things to try on Monday | Rails, key facts in context, validate before acting, turn every hole into a test | Five design checks (`merlino/competency.py`), host-controlled check-in (B3), SHACL gate, paired regression tests | ✓ |
| 36 | Sources | "Merlino: WordLift GitHub repository" | `wordlift/merlino` is **private** | **Blocker:** see below |

## Fixes needed in the deck itself

1. **Slide 26, agenda § numbers.** The deck's slots and topics are kept as they are in
   `docs/WORKSHOP.md`; only the section references change: (§0) stays ·
   (§1–2) stays · "SHACL gate rejects 42.5 km" → (§3–4) · "TempoRun" → (§5) ·
   "Human approval" → (§6–8) · "Gemma 4 on GPU" → (§9–10) · "PROV-O trace, Resolve,
   WoGraph" → (§11).
2. **Slide 32, "traces become training data".** Nothing in the notebook trains a model:
   Experiment 8 changes a *proposal* from recalled evidence. Say "traces can become
   training data" and point to Hippocampus and the GRPO slides, or drop the clause.
3. **Slide 36, repository link.** The repo is private and `docs/RELEASE_APPROVALS.md`
   still has open blockers (Bea's redistribution approval, WordLift licence approval,
   a full GPU rehearsal). Either clear them and make the repo public before
   11 November, or replace the link with "available on request" for the session.

## Wording to keep (Bea's review)

- "SHACL validates and reports violations; Merlin responds by proposing a different
  action." The gate never repairs or chooses 36 km; the agent computes the new candidate
  from the *current* graph-owned limit.
- After pain, runner logging and closure remain possible where the Railroad allows them;
  new training authorization stays blocked until a person takes over.
- Bea's original notebook reported 23/23 tests. Merlino's suite is a different one
  (54 tests, 17 of them in `tests/test_engine.py`): compare coverage, not raw counts.
- The no-rules results come from a disposable synthetic shadow and a scripted agent,
  not from measuring a real model's failure rate.
