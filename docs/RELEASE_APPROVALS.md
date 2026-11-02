# Public release checklist (blocking)

## Licensing and provenance

- [ ] Bea explicitly approves public redistribution, licensing and exact credits for her Map, Railroad and derived teaching artifacts.
- [ ] Confirm that the repository's existing MIT license is compatible with all source contributions; obtain explicit Bea-derived redistribution permission and agree how documentation and teaching assets are licensed.
- [ ] Review confirms no private WoCell, private AOOE methodology tooling, credentials, client data or enterprise policies are in the tree or notebook's embedded archive.
- [ ] Confirm Gemma model usage terms and WoGraph OSS distribution status.

## Verification

- [ ] Run full PySHACL integration tests in a clean Python environment; no skips.
- [ ] Run complete Colab notebook with real Gemma E2B tool calling on the target GPU.
- [ ] Run invalid-tool, approval-bypass and mutation attempts using actual model responses.
- [ ] Complete 2–3 unfamiliar-user rehearsals and record timings, GPU failures and fallbacks.
- [ ] Review and disclose all outbound network payloads, workshop key scope and revocation.

## Branding and release

- [ ] Add **approved, unmodified WordLift logo** and current design-system colors.
- [ ] Use the existing `wordlift/merlino` repository (currently private). Before release, configure Issues, Discussions, topic tags: `aooe`, `shacl`, `rdf`, `knowledge-graph`, `agentic-ai`, `gemma`, `semantic-sovereignty`.
- [ ] Validate the `Open in Colab` button after repository is actually public.
- [ ] Enable required CI and branch protections; verify checks are green on main.
- [ ] Publish as `v0.1-workshop` only when the above release gates pass.

**Important:** A repository being public does not make its contents open source. A compatible explicit license and source rights are required.