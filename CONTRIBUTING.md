# Contributing to Merlino — draft

The purpose of this project is to make ontology-governed AI agents reproducible and understandable.

1. Keep the **propose → validate → authorize → commit** boundary intact; all LLM output is an untrusted proposal.
2. Change controlled meanings in versioned RDF or SKOS, and behavior in reviewed SHACL/BCS. Do not bury policy changes only in prompts.
3. Include negative tests for tool routing, gate outages, state mutations and approvals.
4. Use synthetic data and no embedded API keys. Local/Colab examples must not transmit private athlete data.
5. The default CI suite should run real PySHACL and never require Gemma downloads or paid services.

```bash
python -m pip install -r requirements.txt
python -m pytest -q -ra
```

Licenses and contributor rights are **not yet cleared**. This is staging material, not an open invitation to redistribute unlicensed sources.