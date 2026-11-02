# Merlino + Gemma 4 — Semantic Sovereignty with AOOE

**Workshop draft — do not publish or use personal training data without reviewing the release checklist.**

Merlino is a real tool-using running-coach *agent demo*: Gemma 4 makes proposals, an RDF graph expresses meanings and state, an independent Python **process** runs PySHACL before execution, and a separate runner action is required to approve a plan. The athlete and all fixtures are synthetic.

> **Own the meanings. Govern the decisions. Choose the AI.**

## Start in Colab

1. Upload `notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run the setup cells (they unpack a self-contained reviewed workshop package and install RDFLib / PySHACL).
3. Run the **ontology and SHACL** exercises first; they require no GPU, WordLift credentials or model weights.
4. Choose **Runtime → Change runtime type → T4 GPU** and set `USE_GEMMA = True` in the Gemma cell. Install optional model dependencies, then load `google/gemma-4-E2B-it` (smaller) or E4B (larger, higher memory). A Hugging Face download/access agreement may be needed. No generated answer is substituted if model download or inference fails.
5. Ask: `Propose a 42.5 km week for Alex. What happens when SHACL reviews it?` Try `34.5 km`, then a `consecutive_hard` week. The model can call only four whitelisted tools: read context, propose plan, inspect decision, and hand over to human.
6. Explore the separate human-approval and semantic-edit cells. The model does **not** have these capabilities.

**Colab is NOT on-device or private infrastructure.** The model computes within a Colab VM (not a hosted LLM API), but the VM is Google-managed and downloading model weights needs a network connection. Use a participant-owned machine and trusted RDF services for truly local/sovereign deployment.

## Run with local Python

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python - <<'PY'
from merlino.engine import Merlino
m = Merlino(); print(m.read_history().as_dict())
print(m.propose(42.5).as_dict())  # reject, no plan is saved
print(m.propose(34.5).as_dict())  # eligible for runner review
PY
```

The PySHACL process must be installed and reachable. If it fails, **all actions fail closed**. Unit tests use a clearly named fake gate only for testing the Python lifecycle; SHACL integration tests run only with real PySHACL installed. The separate subprocess demonstrates separation of model and validator, **not** OS-level access isolation: a regulated enterprise must run a privileged/policy-signed gate as an independent protected service.

## From ontology engineering to agentic engineering

| Stage | Artifact | Workshop effect |
| --- | --- | --- |
| **Legend** | Schema.org, SKOS, identifiers and versioned vocabulary | The athlete and workout types are unambiguous |
| **Map** | Bea's `merlin_map.ttl` and competence questions | The model can read relevant training history and the actual meaning of TempoRun |
| **Rulebook** | Bea's Railroad + this project's gate, BCS and audit | A model proposal is not an authorized action; SHACL and explicit approval determine execution |

Bea's source **Map**, **Railroad** and teaching-vocabulary RDF are retained in `ontology/` and `reference/bea/`. The vulnerable original scenario helper is **not** used as an execution engine; see `docs/ARCHITECTURE.md` for changes. The namespace `https://example.org/aooe#` in Bea's original railroad is a **teaching-only runtime namespace**; it is not the official public [AOOE vocabulary](https://w3id.org/aooe#). The public vocabulary covers adoption levels and governance metadata. This package does not claim to pass the full private AOOE methodology's conformance gates.

## Sovereignty boundaries

- **Graph:** RDFLib in the notebook for synthetic inputs; WoGraph OSS is an optional separately deployed backend (not run inside Colab).
- **Model:** Gemma 4 E2B or E4B runs on the participant VM GPU. The exact context packet can be inspected in `Merlino.context()`; there is no external chat inference API.
- **Validator:** the model cannot call arbitrary Python, edit the policy, approve on behalf of the athlete, or replace the SHACL gate. Its tool calls go to a fixed allowlist. The gate is invoked out of process and is fail-closed; stronger process/network permissions are needed in production.
- **Authority:** an explicit `approve_as_runner(human_confirmed=True)` **teaching simulation**, not a cryptographic identity proof, binds plan digest + policy version + meaning revision.
- **Provenance:** decisions and rule messages are serialized as JSON or PROV-O-aligned RDF; the trace is an illustrative local audit, not tamper-proof.
- **Remote services:** WordLift Resolve is **optional, off by default**, receiving only displayed synthetic workout labels with an interactively entered temporary key; WoGraph OSS read-only access requires an explicit HTTPS endpoint and isolated workshop dataset.

## Repo contents

```
notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb  Self-contained workshop notebook
merlino/engine.py                            Model-independent pre-commit engine
merlino/gate_worker.py                       Separate PySHACL process (fail closed)
merlino/gemma_agent.py                       Native Gemma 4 tool use / strict allowlist
merlino/wordlift.py                          Opt-in synthetic-only WordLift Resolve call
merlino/wograph.py                           Opt-in WoGraph OSS read-first adapter
ontology/merlin_map.ttl                     Bea's domain Map (unchanged)
ontology/merlin_railroad.ttl                Bea's Railroad (unchanged)
ontology/policy.ttl                         Versioned fictitious coaching policy
reference/bea/                              Original teaching-only runtime vocabulary/shapes
shapes/sovereign_gate.ttl                   Strengthened SHACL shapes
BCS.md                                      Governed behavior contract
models/                                     None: model weights not redistributed
tests/                                     Negative tests + optional PySHACL integration tests
```

**No model weights, API keys, real training records, production WordLift data, private WoCell code, private methodology tooling or enterprise policy configurations are included.** Review Bea's contribution/licensing and policy release before public distribution.

## Status and test limitations

- Executed local unit tests and SPARQL constraint queries using RDFLib.
- The local authoring container has no PySHACL installed and no network package access. The integration tests explicitly skip if PySHACL is absent; the actual engine refuses to proceed without the validator.
- The authoring container has no Gemma model weights/GPU. **No claim is made that Gemma inference or the full notebook was run end to end.** A T4 Colab rehearsal is required.
- The current WoGraph OSS hosted sandbox is staged for publication; do not assume the repository and launch button are publicly available.

See `docs/FACILITATOR.md` and `docs/RELEASE_CHECKLIST.md` before teaching.
