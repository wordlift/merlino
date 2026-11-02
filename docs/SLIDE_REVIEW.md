# Suggested edits to Bea's CDL 2026 deck

The **original deck is preserved**, not silently changed. The following edits
align its claims with the new `wordlift/merlino` Colab:

| Slide | Current story | Recommended adaptation |
| --- | --- | --- |
| 1–2 | Merlin, 7 cases, 23 tests | Introduce Merlino as WordLift OSS workshop agent; attribute Bea and state that the original kit had 23 tests, whereas the new port has a distinct regression suite. |
| 3 | Unsafe no-rules cases | Keep four headline failures. Clarify that the new implementation records them via a **disposable shadow** with real SHACL inspection, not unsafe external actions or real Gemma misbehavior. |
| 5 | Map vs Railroad | Keep exact 9 states, 10 actions, 2 roles, 3 evidence types; tool list for Gemma is now derived from the Railroad. |
| 8 | 33 km and freshness | Show exact `schema:ExerciseAction` / `mr:distanceKm` SPARQL and the pinned synthetic reference date; do not suggest today's date is magically fresh. |
| 10 | Five CQs | Keep CQ3's exists-path qualification; `run_checks` remains valid. |
| 11–12 | Eleven rules, latest check-in | Preserve G1–G4/B1–B7, and spotlight B3's latest context + pain hold. B2 now sums session distances even if the declared total is false. |
| 14–15 | S1–S7, S5 trace | Keep seven cases and B2/B4/B7 S5 repair. Re-run S5 on the new PROV-O trace. Extra B4 on S7 is fine; G2/G4 are the headline. |
| 16 | 23/23 safety tests | Label **23/23 as Bea's original notebook test run**. Insert the new suite's exact count only after CI passes. |
| 18 | `call_model` stub | Update to optional Gemma 4 with state-filtered tools; note that the original 7 cases remain deterministic. Live GPU validation remains a separate acceptance check. |
| 19 | 34% benchmark | Keep the clear **different task, not Merlino** disclaimer; do not generalize cost figures to Gemma or this workshop. |
| 20 | limitations and CSV | Emphasize Colab is cloud-hosted; use **synthetic** data only. Personal training histories belong in locally controlled infrastructure. |
| 21–22 | take-home | Add 'Own the meanings. Govern the decisions. Choose the AI.' without replacing the Map → Railroad → Gate storyline. |

## Before delivery

- Perform a clean Colab run: install, all 14 scenario episodes, all tests, S5 SPARQL, S4 pain, then Gemma GPU flow.
- Have Bea sign off on source provenance/attribution, policy thresholds and deck text.
- Keep the WordLift key optional and interactive; synthetic-only Resolve queries.
- Offer WoGraph OSS or WordLift managed backend as a separate extension, not a requirement for participants.
