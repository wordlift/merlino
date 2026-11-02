# Privacy, security, and the sovereignty boundary

**Educational only.** The fictional athlete and synthetic rules are not real-world training advice.

- **Colab is cloud compute.** Using Gemma inside a Colab VM avoids an external chat-model API, but does not make the workload on-device or private from the cloud platform. Never upload confidential or actual athlete/health data.
- **The model is untrusted.** Only whitelisted tools can enter the execution engine; all candidates must pass the independent SHACL gate before committing state.
- **A subprocess is not OS security isolation.** Production SHACL must be hosted with independent access controls and signed/versioned policies.
- **Approval is simulated.** A boolean is not identity attestation; production needs authenticated actors and approval provenance.
- **Meaning is an authority.** Reclassifying workout types can change agent decisions. In enterprise systems, only authorized stewards may publish semantic/policy changes.
- **Resolve is remote.** The optional integration sends only a fixed synthetic workout-label payload when users opt in; temporary workshop keys must be scoped, short-lived, quota-limited and revocable, and never embedded in files.
- **WoGraph OSS is separately hosted.** Remote graph use is opt-in and should target an isolated dataset. Verify URL, scope and write permissions independently.

Before any public release, scan all included files and notebook embedded archives for secrets, licensing issues, model output artifacts and customer data.
