"""Isolated validation process: stdin JSON candidate graph, stdout JSON verdict.

External to the model process, but not a secure enterprise sandbox. Only an independent,
privileged service/process with controlled filesystem/IAM qualifies as a security boundary.
"""
import json
import sys
from pathlib import Path
from rdflib import Graph, RDF, Namespace

SH = Namespace('http://www.w3.org/ns/shacl#')
ROOT = Path(__file__).resolve().parent.parent


def validate_candidate(candidate_ttl: str) -> dict:
    from pyshacl import validate  # fail-closed if dependency absent
    graph = Graph().parse(data=candidate_ttl, format='turtle')
    shapes = Graph().parse(ROOT / 'shapes' / 'sovereign_gate.ttl', format='turtle')
    conforms, report, _ = validate(graph, shacl_graph=shapes, inference='none',
                                   advanced=True, allow_warnings=True, abort_on_first=False)
    if not isinstance(report, Graph):
        return {'conforms': False, 'violations': ['SHACL_VALIDATION_FAILURE: ' + str(report)],
                'warnings': [], 'gate': 'pyshacl validation failure', 'candidate_triples': len(graph)}
    violations, warnings = [], []
    for r in report.subjects(RDF.type, SH.ValidationResult):
        msg = str(report.value(r, SH.resultMessage) or 'Undescribed validation failure')
        severity = report.value(r, SH.resultSeverity)
        (warnings if severity == SH.Warning else violations).append(msg)
    return {'conforms': bool(conforms) and not violations,
            'violations': sorted(set(violations)), 'warnings': sorted(set(warnings)),
            'gate': 'pyshacl subprocess', 'candidate_triples': len(graph)}


if __name__ == '__main__':
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload.get('candidate_ttl'), str) or len(payload['candidate_ttl']) > 250_000:
            raise ValueError('Candidate missing or too large')
        print(json.dumps(validate_candidate(payload['candidate_ttl'])))
    except Exception as exc:
        print(json.dumps({'conforms': False, 'violations': ['GATE_UNAVAILABLE: ' + type(exc).__name__ + ': ' + str(exc)],
                          'warnings': [], 'gate': 'unavailable'}))
        sys.exit(1)
