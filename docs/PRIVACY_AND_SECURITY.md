# Privacy, security, and the sovereignty boundary

**Educational only.** The fictional athlete and synthetic policies are not real-world training advice.

- **Colab is cloud compute.** Gemma inside a Colab VM avoids an external chat-model API but is not on-device or confidential computing. Use synthetic data only.
- **Model output is untrusted.** Only allowlisted tools can reach the execution engine. Candidates must pass independent SHACL validation before writes.
- **The validator subprocess is not a security boundary.** Production deployments require independently controlled policies, authenticated APIs, and protected execution.
- **Human approval is simulated.** The workshop's boolean approval is not production identity attestation.
- **Semantic governance matters.** Only authorized stewards should alter ontology classifications and publish new policy revisions.
- **WordLift Resolve is remote.** Opt-in calls must show the exact outbound synthetic data; temporary workshop keys must be scoped, revocable, rate-limited, and never committed.
- **WoGraph OSS remains optional.** Only isolated datasets on explicitly trusted endpoints should be used.

Before publishing: verify source rights and license compatibility, scan for secrets, execute full PySHACL/Gemma/Colab acceptance tests, and review outbound data flows.
