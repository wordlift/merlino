"""Workshop-only semantic memory: an RDF world plus remembered, governed decisions.

The TriG file is a trusted *local teaching artifact*, not an authenticated
multi-user graph store. The model has no tool to read arbitrary paths or write
memory. State and approval are deliberately NOT restored from the file.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from uuid import uuid4

from rdflib import Dataset, Graph, Literal, URIRef
from rdflib.namespace import PROV, RDF, XSD

from .history import history_summary

BASE = 'https://example.org/merlino-memory/'
ROOT = URIRef(BASE + 'memory')
WORLD = URIRef(BASE + 'world')
POLICY = URIRef(BASE + 'policy')
AUDIT = URIRef(BASE + 'audit')
META = URIRef(BASE + 'metadata')
FORMAT = URIRef(BASE + 'formatVersion')
REFERENCE_DATE = URIRef(BASE + 'referenceDate')
VERSION = 'merlino-episode-v1'
MAX_BYTES = 2_000_000


def _copy(source: Graph, target: Graph):
    for triple in source:
        target.add(triple)


def _scoped_audit(source: Graph) -> Graph:
    """Mint unique IRIs on export so each resumed episode stays independent."""
    from .engine import TRACE
    session = BASE + 'episode-' + uuid4().hex + '/'
    result = Graph()
    def scoped(term):
        if isinstance(term, URIRef) and str(term).startswith(str(TRACE)):
            return URIRef(session + str(term)[len(str(TRACE)):])
        return term
    for subject, predicate, obj in source:
        result.add((scoped(subject), predicate, scoped(obj)))
    return result


def save_memory(engine, path: str | Path) -> Path:
    """Persist ontology + synthetic runner world + policy + PROV-O audit as TriG.

    This is an explicit facilitator-controlled commit, never a hidden model write.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset()
    _copy(engine.map, dataset.graph(WORLD))
    _copy(engine.policy, dataset.graph(POLICY))
    _copy(engine.memory_trace, dataset.graph(AUDIT))
    _copy(_scoped_audit(engine.provenance), dataset.graph(AUDIT))
    metadata = dataset.graph(META)
    metadata.add((ROOT, FORMAT, Literal(VERSION)))
    metadata.add((ROOT, REFERENCE_DATE, Literal(engine.today.isoformat(), datatype=XSD.date)))
    payload = dataset.serialize(format='trig').encode('utf-8')
    if len(payload) > MAX_BYTES:
        raise ValueError('Synthetic memory file exceeds the workshop size limit')
    descriptor, temp = tempfile.mkstemp(prefix='.merlino-', suffix='.trig', dir=str(dest.parent))
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, dest)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return dest


def load_memory(engine, path: str | Path) -> dict:
    """Load trusted local RDF memory, but never reinstate a session's authority.

    A fresh Merlino session remains at Start. Approvals, proposals, agent state,
    human confirmations, and tool permissions are not imported from this file.
    """
    from .engine import MR
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_BYTES:
        raise ValueError('Missing or oversized workshop memory file')
    dataset = Dataset().parse(source, format='trig')
    meta = dataset.graph(META)
    if (ROOT, FORMAT, Literal(VERSION)) not in meta:
        raise ValueError('Incompatible semantic memory version')
    reference = meta.value(ROOT, REFERENCE_DATE)
    reference_day = date.fromisoformat(str(reference))
    world, policy, audit = (dataset.graph(WORLD), dataset.graph(POLICY), dataset.graph(AUDIT))
    if not world or not policy:
        raise ValueError('Semantic world and policy must both be present')
    restored = Graph()
    _copy(world, restored)
    restored_policy = Graph()
    _copy(policy, restored_policy)
    engine.map = restored
    engine.policy = restored_policy
    engine.memory_trace = Graph()
    _copy(audit, engine.memory_trace)
    facts = history_summary(engine.map, reference_day)
    engine.baseline_km = facts['baseline_km']
    engine.history_age_days = facts['age_days']
    engine.today = reference_day
    engine.source_history = facts
    # Explicitly fail closed on authority replay (including after a future code change).
    engine.state = MR.Start
    engine.history_present = False
    engine.approval = None
    engine.checked_proposals.clear()
    engine.evidence.clear()
    engine.plan = None
    engine.started = False
    return {'world_triples': len(engine.map), 'prior_decisions': len(recall_memory(engine.memory_trace, limit=1000)),
            'policy_version': engine.policy_version, 'meaning_revision': engine.meaning_revision,
            'state': engine.state_name, 'approval_restored': False}


def recall_memory(graph: Graph, *, limit: int = 5) -> list[dict]:
    """Retrieve prior actions and validation findings from the RDF audit."""
    from .engine import MG
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError('Invalid memory recall limit')
    rows = []
    for decision in graph.subjects(RDF.type, PROV.Activity):
        action = graph.value(decision, MG.actionName)
        accepted = graph.value(decision, MG.accepted)
        if action is None or accepted is None:
            continue  # only decision nodes, not their execution/validation activities
        rows.append({'decision_iri': str(decision), 'action': str(action),
                     'accepted': bool(accepted.toPython()),
                     'violations': sorted(str(v) for v in graph.objects(decision, MG.violation)),
                     'policy_version': str(graph.value(decision, MG.policyVersion) or ''),
                     'meaning_revision': str(graph.value(decision, MG.meaningRevision) or '')})
    return sorted(rows, key=lambda x: x['decision_iri'])[-limit:]
