"""Design-time checks: competency questions as regression tests over the MERLIN RAILROAD itself.

Run-time shapes check one proposed step. These queries check the state space before any
agent runs: they return counterexamples (empty result = the property holds).
All 5 queries are generic across AOOE railroads.
"""
from __future__ import annotations

from rdflib import Graph
from rdflib import Namespace
from rdflib.namespace import RDF, SKOS
PREFIXES = {'aooe': Namespace('https://example.org/aooe#'),
            'mr': Namespace('https://example.org/merlin-railroad#'),
            'rdf': RDF, 'skos': SKOS}

STEP = "(^aooe:fromState/aooe:toState)"     # one move, as a SPARQL property path: state -> state

CQS = {
    "CQ1 no trap states": (
        "Every state reachable from the initial state can still reach a terminal state.",
        f"""SELECT DISTINCT ?bad WHERE {{
              ?init aooe:isInitial true .
              ?init {STEP}* ?bad .
              FILTER NOT EXISTS {{ ?bad aooe:isTerminal true }}
              FILTER NOT EXISTS {{ ?bad {STEP}* ?t . ?t aooe:isTerminal true }}
            }}"""),
    "CQ2 irreversible needs a human": (
        "Every irreversible action requires evidence that only a human role can produce.",
        """SELECT DISTINCT ?bad WHERE {
              ?bad a aooe:Action ; aooe:isIrreversible true .
              FILTER NOT EXISTS {
                ?bad aooe:requiresEvidence ?e .
                ?p aooe:producesEvidence ?e ; aooe:allowedRole ?r . ?r a aooe:HumanRole .
                FILTER NOT EXISTS { ?p aooe:allowedRole ?r2 . FILTER NOT EXISTS { ?r2 a aooe:HumanRole } }
              }
            }"""),
    "CQ3 evidence obtainable in time": (
        "For every move whose action requires evidence E, some path from the initial state passes "
        "through an action producing E before reaching the move (exists-path, not all-paths).",
        f"""SELECT DISTINCT ?bad ?e ?from WHERE {{
              ?t aooe:fromState ?from ; aooe:viaAction ?bad . ?bad aooe:requiresEvidence ?e .
              FILTER NOT EXISTS {{
                ?init aooe:isInitial true .
                ?pt aooe:viaAction ?p ; aooe:fromState ?s1 ; aooe:toState ?s2 .
                ?p aooe:producesEvidence ?e .
                ?init {STEP}* ?s1 .
                ?s2 {STEP}* ?from .
              }}
            }}"""),
    "CQ4 no orphan tools": (
        "Every action appears in at least one transition (otherwise it is a tool no state exposes).",
        """SELECT DISTINCT ?bad WHERE {
              ?bad a aooe:Action .
              FILTER NOT EXISTS { ?t aooe:viaAction ?bad }
            }"""),
    "CQ5 agent-readable definitions": (
        "Every state and action carries a non-empty skos:definition (the model reads it).",
        """SELECT DISTINCT ?bad WHERE {
              { ?bad a aooe:State } UNION { ?bad a aooe:Action }
              FILTER NOT EXISTS { ?bad skos:definition ?d . FILTER (STRLEN(STR(?d)) > 0) }
            }"""),
}


def run_checks(railroad: Graph, only: list[str] | None = None) -> dict:
    """Returns {cq: (statement, [counterexample rows])}. Empty list = passes."""
    out = {}
    for name, (statement, q) in CQS.items():
        if only and name not in only:
            continue
        rows = [tuple(str(v).split("#")[-1] if v is not None else None for v in r)
                for r in railroad.query(q, initNs=PREFIXES)]
        out[name] = (statement, sorted(set(rows)))
    return out


def report(railroad: Graph) -> bool:
    """Prints readable CQ audit report. Returns True if all pass."""
    all_passed = True
    results = run_checks(railroad)
    for name, (statement, rows) in results.items():
        status = "PASS" if not rows else "FAIL"
        if rows:
            all_passed = False
        print(f"[{status}] {name}: {statement}")
        for r in rows:
            print("       counterexample:", ", ".join(x for x in r if x))
    return all_passed
