<div align="center">

### WordLift Lab · AOOE in Practice

# MERLINO 🏃

**An ontology-governed AI running coach.**

**Own the meanings. Govern the decisions. Choose the AI.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#quickstart)
[![Semantic Map](https://img.shields.io/badge/Semantics-RDF%20%2B%20SKOS-2669A6)](ontology/)
[![Governance](https://img.shields.io/badge/Governance-SHACL-276579)](shapes/)
[![Model](https://img.shields.io/badge/LLM-Gemma%204-6574B7)](requirements-gemma.txt)
[![Status](https://img.shields.io/badge/Status-Preview-orange)](docs/RELEASE_APPROVALS.md)

*A hands-on reference implementation for **Agent-Oriented Ontology Engineering (AOOE)**.*

*Workshop for **From Map to Railroad**, the Connected Data London 2026 masterclass by Beatrice Gamba and Andrea Volpini, 11 November 2026. [Agenda](docs/WORKSHOP.md) · [deck alignment](docs/SLIDE_REVIEW.md)*

</div>

> **Pre-publication draft.** Bea's redistribution permission, WordLift licensing approval, and full Gemma/Colab/PySHACL integration checks are pending. Do not publish before [release approvals](docs/RELEASE_APPROVALS.md) are completed.

## Why Merlino?

An organization may own a powerful LLM and still lose control of its **meanings**: what counts as reliable evidence, a safe action, or an authorized decision. A prompt cannot establish that control.

**Merlino makes the boundary visible.** Gemma 4 plays the conversational running coach for a **fictional athlete**. RDF and SKOS describe the domain. A versioned Rulebook and **independent SHACL gate** validate candidate actions. Human approval is a separate step. The agent cannot rewrite its own rules or award itself permission.

This demonstrates **Semantic Sovereignty**: organizations control semantics, evidence, decisions, and data access even when language models change.

## The knowledge graph is the agent's world and persistent semantic memory

Merlino does not merely *look up* RDF. Its ontology and graph define the world
it inhabits, including the runner's observations, coaching meanings, the
Railroad, policies, and past decision traces. **The SHACL gate validates; the
agent, not the gate, decides how to respond to validation results.**

The workshop now includes a local RDF TriG persistence exercise: trigger a
rejected plan, save the graph and trace, create a fresh Merlino, and retrieve
that decision without restoring old approval. See
[semantic memory and research extensions](docs/SEMANTIC_MEMORY.md).

## One architecture, four responsibilities

```mermaid
flowchart TD
    U[Runner's request] --> G[Gemma: propose and explain]
    G --> C[Candidate tool action: no side effects]
    M[(RDF + SKOS: owned meanings)] --> C
    C --> S{Independent SHACL gate}
    P[(Versioned policy + Railroad)] --> S
    S -->|Reject| R[Agent explains the violation and proposes again]
    R --> G
    S -->|Eligible| H{Human confirmation}
    H -->|No| N[No action]
    H -->|Yes + revalidate| A[Commit governed state]
    S --> T[(Decision trace)]
    A --> T
```

**Gemma proposes · RDF defines · SHACL validates · humans authorize.** The model proposes; the graph disposes.

**Experiment 8 (Hippocampus-inspired, not model training):** compare two new
Merlino sessions with the same graph and current policy, but only one can
read an earlier B2 rejection. The scripted agent uses that episode as evidence
to choose a fresh proposal; SHACL validates both independently. For optional
real Gemma inference, a bounded `recall_governed_experience` tool is available
only in the explicit memory-enabled condition. [Full method](docs/SEMANTIC_MEMORY.md).


## Quickstart

### 1. Guided notebook (CPU-first, Gemma optional)

Open [`notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb`](notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb) in [Google Colab](https://colab.research.google.com/) via its notebook upload or GitHub-import flow. The notebook includes its own educational source bundle.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wordlift/merlino/blob/main/notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb)

**Colab is a Google-managed VM, not an on-device or private deployment. Use only synthetic data.** No hosted LLM API is required for the CPU exercises. The optional Gemma model runs in the notebook runtime and requires a suitable GPU and model access agreement.

### 2. Python: real independent SHACL gate

```bash
python -m pip install -r requirements.txt
python -m pytest -q -ra
python -c 'from merlino.engine import Merlino; m=Merlino(); print(m.read_history().as_dict()); print(m.propose(42.5).as_dict())'
```

A missing validator makes the real engine fail closed. Tests explicitly identify any fake gate; no model or SHACL successes are fabricated.

## Three interactive experiments

1. **Describe meanings:** inspect Bea's [Map](ontology/merlin_map.ttl), the Railroad and the fictional athlete. How does `TempoRun` differ from an untyped text label?
2. **Govern actions:** request a **42.5 km** week for a runner with a **33 km** baseline; inspect the SHACL rejection and audit trace.
3. **Change meaning, not the prompt:** change whether `TempoRun` is considered a hard session; rerun the consecutive-hard-days scenario and inspect the changed result.

**The synthetic mileage policy is a teaching device, not running or medical advice.**

## From traditional ontology engineering to AOOE

Each level adds a right; only the Rulebook grants agency. **Legend ⊂ Map ⊂ Rulebook**: each level contains the one before.

| Level | Right | Main deliverable | Merlino example |
| --- | --- | --- | --- |
| **Legend** | Addressability | Vocabulary, identity and pinned meanings | Workout terms and identifiers |
| **Map** | Readability | Classes, properties and competency questions | Athlete, sessions, plan, evidence |
| **Rulebook** | Agency | Shapes, behaviour specs, approval and trace | The Railroad, the SHACL gate, human sign-off |

The levels are defined in the public AOOE vocabulary, [`w3id.org/aooe#`](https://w3id.org/aooe#) ([source](https://github.com/wordlift/aooe)). This educational prototype does **not** claim conformance to WordLift's private AOOE methodology gates.

## Semantic sovereignty and disclosure

| Component | Workshop default | Enterprise-sovereign version |
| --- | --- | --- |
| Model | Gemma 4 on Colab GPU (optional) | Privately hosted Gemma / replaceable model |
| Meaning + state | RDFLib, local to the notebook VM | Versioned WoGraph OSS deployment |
| Validation | PySHACL in a separate subprocess | Authenticated protected policy service |
| Approval | Explicit simulation | Verified identity and authorization |
| External services | Off by default | Scoped, audited WordLift Resolve / graph API |

WordLift Resolve sends only fixed synthetic labels when enabled and a temporary credential is supplied. No model weights, live credentials, personal histories, private WoCell code or private lab methodology are bundled.

## Explore

**KG Atlas:** download [`docs/merlino-kg-atlas.html`](docs/merlino-kg-atlas.html) and open it in a browser. It is a 3D map of the Legend, Map and Rulebook with walk-throughs of Bea's seven scenarios, semantic memory (Experiment 8) and the Gemma loop. It works offline (three.js is inlined) and describes the workshop Colab's code.

[Workshop agenda](docs/WORKSHOP.md) · [KG Atlas (interactive, offline HTML)](docs/merlino-kg-atlas.html) · [Facilitator's guide](docs/FACILITATOR.md) · [Architecture](docs/ARCHITECTURE.md) · [Behavioral specification](BCS.md) · [Privacy and security](docs/PRIVACY_AND_SECURITY.md)

## Credits and licensing

An educational WordLift Lab prototype based on **Bea's Merlin Running Pal** ontology and Railroad, extended with an independent gate, Gemma agent adapter, and sovereignty exercises. [Source notices](NOTICE.md) preserve the pending attribution and licensing review.

The [AOOE vocabulary](https://github.com/wordlift/aooe) is a separate CC BY 4.0 project. [WoGraph OSS](https://github.com/wordlift/wograph-oss) is a separately deployed runtime. Gemma model weights are not included and use Google's applicable terms.

**This staging package has no public open-source license yet.** Publication and licensing require explicit contributor/WordLift approval. See [release checklist](docs/RELEASE_APPROVALS.md).

## CDL 2026: all seven of Bea's scenarios

The merged workshop source now includes a **full nine-state, ten-action** synthetic
lifecycle and two reproducible conditions for each of Bea's cases:

- **Governed:** a real independent PySHACL gate validates every RDF candidate before
  actions are committed; invalid proposals cannot alter the runner state.
- **No-rules shadow:** a disposable in-memory teaching simulation enforces nothing,
  but still runs the validator as a *shadow* to reveal each failure. No data or
  action leaves the synthetic sandbox. This is not a real-model failure-rate test.

See [the scenario matrix and slide-by-slide alignment](docs/BEA_CASE_COVERAGE.md)
for S1 normal week, S2 overload, S3 stale data, S4 pain escalation, S5 changed
plan, S6 consecutive hard sessions and S7 skipped approval. The new notebook
includes all paired runs, a detailed S5 PROV-O replay, S4 pain context, and
an executable regression suite.

**Model distinction:** Gemma 4 is optional *live* agent inference, with tools
filtered by the RDF Railroad. The seven scenario metrics come from deterministic
scripted replay, **not** from Gemma sampling. Running Gemma in Colab uses a
Google-hosted VM; that is not sovereign local deployment for private user data.
