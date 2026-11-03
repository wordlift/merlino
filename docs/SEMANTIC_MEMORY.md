# The KG is Merlino’s environment and memory

Merlino **inhabits** a semantic environment: its ontology defines meanings, its
RDF facts describe the synthetic athlete’s world, the Railroad describes legal
transitions, and its RDF provenance captures what was observed, proposed,
validated and executed. The SHACL gate **validates; it does not repair**.

## Observable workshop loop

**Remember → Navigate/Interpret → Propose → Validate → Act → Remember.**
The model or scripted agent may inspect evidence and propose a new candidate
when a validation message is returned. Only authorized changes enter the world.

## Persistence exercise (synthetic data only)

```python
from merlino.engine import Merlino
from pathlib import Path

file = Path('synthetic_merlino_memory.trig')
a = Merlino()
a.read_history()
a.propose(42.5)               # B2 reject, documented in RDF provenance
a.save_memory(file)             # facilitator-controlled local TriG export

b = Merlino()                  # fresh agent instance, no prior Python state
summary = b.load_memory(file, facilitator_confirmed=True)
print(summary)
print(b.recall_memory())       # prior rejection and policy revision from RDF
assert b.state_name == 'Start' and b.approval is None
```

The file holds named RDF graphs for world, policy, and audit. A fresh session
retrieves the prior decision without reinstating approval, tool permissions,
checked proposals, or active state. This is a **bounded educational graph
snapshot**, not a hardened memory service: workshop local files are not
signed, authenticated or tamper-evident. External inference is not required.

## Connection to WordLift research

- **AOOE / Railroad:** machine-readable identity, authority, validation and actions.
- **RLM-on-KG:** adaptive, multi-hop navigation of an RDF graph to find evidence;
  [public research](https://github.com/wordlift/rlm-on-kg). Not implemented in this demo.
- **Hippocampus:** research on learning reliable KG navigation from verified
  experience and training traces. This demo does **not** claim online learning,
  checkpoint training, or the Hippocampus verifier.
- **WoGraph OSS:** a potential owned, persistent graph backend. This demo uses
  RDFLib and a local TriG export; no managed graph persistence is implied.

**Semantic sovereignty** is control over meanings, evidence, memory, policies and
which actions are authorized. Colab is still cloud-hosted; do not upload private
training or medical data to this teaching environment.
