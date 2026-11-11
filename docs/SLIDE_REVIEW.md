# Connected Data London 2026 — live deck ↔ Merlino notebook alignment

**Deck:** [From Map to Railroad — Gamba & Volpini](https://docs.google.com/presentation/d/1QRu-PWEhXNTFcV1Iw0b9E1lMlKV4Xa5oxwKLD-VKXp0/edit), Connected Data London, 11 November 2026. **40 slides** as inspected on 11 November 2026. Earlier review of a 37-slide version is superseded by this mapping.

**Notebook:** [Merlino Gemma 4 AOOE Workshop](../notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb), cells 0–42, §0–§11. The default CPU route is scripted and uses real independent PySHACL, **not Gemma inference**. The optional Gemma 4 path requires a Colab GPU, model access and `USE_GEMMA=True`.

**Reproducibility boundary:** Alex's records are fictional and are bundled in the Colab notebook. Bea's private run results in the deck come from a **separate local execution** and cannot be reproduced from the public Colab alone. Do not upload real running or health histories to hosted Colab.

## Slide-by-slide review: workshop slides 26–36

| Slide | Live deck claim | Notebook correspondence | Review |
| --- | --- | --- | --- |
| 26 · Agenda | §0–§11, cells 0–42, seven timed slots | §0 setup 0–3 · §1–2 Map/Railroad 4–11 · §3–4 Gate/cases 12–21 · §5–6 Meaning/approval 22–25 · §7–8 Memory 26–29 · §9–10 Gemma 30–37 · §11 Trace/opt-ins 38–42 | **Aligned:** section and cell ranges. The timing is a presentation plan, not an execution requirement. |
| 27 · Two runners | Alex (fictional, Colab) vs Bea (local/private) | Alex's synthetic baseline and eight runs in §1; Bea's 339-run, 31.36 km comparison is from separate local work | **Clarify:** no real Bea data is bundled or used in Colab. |
| 28 · Map | RDF facts, hard-session meaning, computed baseline | §1, cells 5 and 7 (Alex: 33 km and two-day synthetic freshness) | **Aligned for Alex.** Bea's 1,736 triples and private run totals are external results, not cell 5 or 7 output in Colab. |
| 29 · Railroad | Five design checks, nine states, ten actions, two roles | §2, cells 9 and 11; `merlino/competency.py` | **Aligned** for shared ontology/Railroad. |
| 30 · Governed lifecycle | Propose, SHACL validate, human approve, PROV-O record | §2–4, cells 8–21; `ontology/merlin_railroad.ttl` | **Wording caution:** the scripted agent, not Gemma, is used by the default CPU exercises. |
| 31 · Gate | Fictional B2 +10% limit blocks 42.5 km; 34.5 km is eligible for Alex only | §3, cells 12–13 | **Aligned** for Alex. Bea's 34.496 km limit is from separate local execution. |
| 32 · Seven scenarios | S1–S7 no-rules shadow vs governed replay; B1/B2/B3/B4/B5/B7/G2/G4 | §4, cells 14–21; `merlino/scenarios.py` | **Aligned:** scripted, not Gemma. Full 54-test CPU CI confirmed on main on 11 Nov 2026. Bea values separately sourced. |
| 33 · Change a meaning and approve | `TempoRun` hard classification changes B5; human authorization stays separate | §5–6, cells 22–25 | **Aligned for Alex.** Simulated consent is not authenticated production authority. |
| 34 · Governed memory | TriG persists an earlier rejection; recall does not restore approval; 1.05 policy compare | §7–8, cells 26–29 | **Aligned for Alex:** scripted experiment is an example of evidence-guided proposals, not training or guaranteed Gemma improvement. Bea results external. |
| 35 · Gemma and decision diary | Gemma 4 E2B optional on T4; provenance/remote services opt-in | §9–11, cells 30–41 | **Aligned:** `USE_GEMMA=False` by default; Resolve/WoGraph off by default. |
| 36 · Atlas | 3D KG atlas; 66 nodes, 19 rules, S1–S7/memory/Gemma tours | `docs/merlino-kg-atlas.html` (model stats: 66 nodes, 19 rules) | **Fix live slide:** use the [Pages Atlas](https://wordlift.github.io/merlino/) link instead of a relative HTML path or `wor.ai` redirect. |

## Recommended deck edits (not in this PR)

1. **Slide 30:** replace “Merlino proposes (Gemma 4)” with “Agent proposes (scripted by default; Gemma optional).” The gate independently validates and never repairs.
2. **Slides 27–34:** add a compact provenance note distinguishing Alex's public Colab data from Bea's private local execution.
3. **Slide 26:** link “COLAB NOTEBOOK” to [the runnable notebook](https://colab.research.google.com/github/wordlift/merlino/blob/main/notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb).
4. **Slide 36:** update “OPEN THE ATLAS” to [GitHub Pages](https://wordlift.github.io/merlino/) and remove the old relative-path instruction. The Pages deploy workflow completed successfully on 11 Nov 2026.

## Scientific and release safeguards

- The gate validates and reports violations; the scripted agent or Gemma interprets codes and **proposes** any repair. SHACL does not itself choose 36 km.
- Memory can change a *new proposal*, not rewrite model weights or reinstate human approval.
- The no-rules shadow is an isolated teaching simulation, **not** measured real-model failure rates.
- Bea's original 23 tests and Merlino's separate 54 tests must not be represented as directly comparable model benchmarks.
- CI on the default CPU path is green; full clean-GPU inference tests, contributor consent and licensing approvals remain distinct release gates. See [Release approvals](RELEASE_APPROVALS.md).
