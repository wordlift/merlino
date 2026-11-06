# Public release checklist (blocking)

## Licensing and provenance

- [ ] Bea explicitly approves public redistribution, licensing and exact credits for her Map, Railroad and derived teaching artifacts.
- [ ] WordLift approves the code and notebook license (proposal: Apache-2.0 for new code, subject to provenance/compatibility) and documentation license (proposal: CC BY 4.0).
- [ ] Review confirms no private WoCell, private AOOE methodology tooling, credentials, client data or enterprise policies are in the tree or notebook's embedded archive.
- [ ] Confirm Gemma model usage terms and WoGraph OSS distribution status.

## Verification

- [x] Run full PySHACL integration tests in a clean Python environment; no skips. (CI on Python 3.11–3.13: 54 tests, plus the same tests on the notebook's embedded bundle and a full CPU notebook run.)
- [ ] Run complete Colab notebook with real Gemma E2B tool calling on the target GPU.
- [ ] Run invalid-tool, approval-bypass and mutation attempts using actual model responses.
- [ ] Complete 2–3 unfamiliar-user rehearsals and record timings, GPU failures and fallbacks.
- [ ] Review and disclose all outbound network payloads, workshop key scope and revocation.

## Branding and release

- [ ] Add **approved, unmodified WordLift logo** and current design-system colors.
- [ ] Open `wordlift/merlino` with `main`, Issues, Discussions, topic tags: `aooe`, `shacl`, `rdf`, `knowledge-graph`, `agentic-ai`, `gemma`, `semantic-sovereignty`.
- [ ] Validate the `Open in Colab` button after repository is actually public.
- [ ] **Before 11 Nov 2026:** the final CDL deck (sources slide) links this repository. Either clear the blockers above and make it public before the session, or change the deck link (see `docs/SLIDE_REVIEW.md`).
- [ ] Enable required CI and branch protections; verify checks are green on main.
- [ ] Publish as `v0.1-workshop` only when the above release gates pass.

**Important:** A repository being public does not make its contents open source. A compatible explicit license and source rights are required.
