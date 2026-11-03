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

## Experiment 8 — Experience → Recall → Reinterpret → Validate

This is a **memory-access ablation**, not a claim that the LLM learned new
weights. It is deliberately separate from Bea's seven safety scenarios.

1. Generate a B2 rejection for a 42.5 km plan under the synthetic v1 policy,
   save the RDF world and decision provenance, and start two fresh sessions.
2. Give both sessions the **same** graph, runner baseline (33.0 km), new 1.05
   progression policy, allowed actions and SHACL gate.
3. The **memory-withheld** scripted policy attempts the original 42.5 km.
   The **memory-available** scripted policy reads the recorded B2 failure,
   then computes a new candidate from the **current** policy: 34.6 km, rather
   than copying a stale 36 km correction.
4. SHACL independently validates each *agent-produced* candidate. The
   experiment records first-pass acceptance; it does not grant human approval.

```python
from merlino.episodic_experiment import compare_recall
report = compare_recall('synthetic_merlino_memory.trig', ratio=1.05)
for arm in report['arms']:
    print(arm['condition'], arm['first_proposal_km'],
          arm['first_pass_accepted'], arm['memory_used'])
```

**Fair comparison:** both arms have the same KG world, current policy and
model/task; only access to prior episode records differs. It is a scripted
agent policy, so the expected outcome illustrates a *mechanism*, not empirical
model generalization or learning. There is no retrieval from a remote provider.

**Optional live Gemma follow-up:** the `MerlinoAgent(..., memory_access=True)`
advertises `recall_governed_experience`, a bounded read-only RDF tool. The
control `memory_access=False` does not advertise it and explicitly denies a
forged call. `get_training_context` does not leak earlier episodes in either
condition. Execute each arm with the same Gemma weights and prompt and report
actual tool calls, validation results and proposal quality. A model may ignore
memory or infer policy from current context; the outcome is not guaranteed.

AOOE owns and validates the environment. RLM-on-KG-style navigation and
Hippocampus-style learning *from verified traces* remain **separate research
projects** until integrated and evaluated here. No gates generate repairs.
