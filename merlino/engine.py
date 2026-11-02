"""Model-independent, fail-closed Merlino action engine.

Bea's RDF Map/Railroad are the semantic source. Model outputs cannot mutate
athlete, ontology, policy, or approval. Candidate -> SHACL subprocess -> commit.
All athlete observations are SYNTHETIC training fixtures.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, PROV, XSD

ROOT = Path(__file__).resolve().parent.parent
MR = Namespace('https://example.org/merlin-railroad#')
OPS = Namespace('https://example.org/aooe#')  # Bea's teaching-only runtime namespace, NOT the public AOOE vocabulary
MG = Namespace('https://example.org/merlino-gemma#')
EX = Namespace('https://example.org/kg/')
TRACE = Namespace('https://example.org/merlino-traces/')


@dataclass(frozen=True)
class Session:
    day: int
    km: Decimal
    kind: URIRef


@dataclass(frozen=True)
class Plan:
    sessions: tuple[Session, ...]
    planned_km: Decimal

    def digest(self) -> str:
        data = {'km': str(self.planned_km), 'sessions':
                [{'day': x.day, 'km': str(x.km), 'kind': str(x.kind)} for x in self.sessions]}
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Approval:
    digest: str
    policy_version: str
    meaning_revision: str
    actor: str = 'runner-confirmed-demo'


@dataclass
class Decision:
    action: str
    accepted: bool
    committed: bool
    from_state: str
    to_state: str
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gate: str = ''
    candidate_hash: str = ''

    def as_dict(self):
        return vars(self).copy()


def graph_hash(g: Graph) -> str:
    # Only fixed non-blank-node teaching vocabularies are used in this hash.
    entries = sorted(' '.join(str(term.n3()) for term in triple) for triple in g)
    return hashlib.sha256('\n'.join(entries).encode()).hexdigest()[:16]


def example_plan(km: float = 34.5, style: str = 'normal') -> Plan:
    """Provide a reproducible proposal candidate; no mutation and no approval.

    The model chooses the total and style. Session distribution is a deterministic
    workshop skeleton, *not* a professional personalized training plan.
    """
    if style not in ('normal', 'consecutive_hard'):
        raise ValueError('Unknown style')
    if not 1 <= float(km) <= 80 or not Decimal(str(km)).is_finite():
        raise ValueError('Synthetic proposed total must be 1–80 km')
    total = Decimal(str(km)).quantize(Decimal('0.1'))
    # Allocate variation to Sunday; preserve seven complete days and exact sums.
    base = (
        [(0, MR.RestDay), (7, MR.EasyRun), (0, MR.RestDay), (6.5, MR.Intervals),
         (0, MR.RestDay), (6, MR.EasyRun), (15, MR.LongRun)]
        if style == 'normal' else
        [(0, MR.RestDay), (6, MR.EasyRun), (7, MR.Intervals), (7.5, MR.TempoRun),
         (0, MR.RestDay), (0, MR.RestDay), (14, MR.LongRun)]
    )
    fixed = sum(Decimal(str(n)) for n, _ in base[:-1])
    if total < fixed:
        raise ValueError('Distance too small for this fixed teaching skeleton')
    sessions = tuple(Session(i + 1, (total - fixed) if i == 6 else Decimal(str(distance)), kind)
                     for i, (distance, kind) in enumerate(base))
    return Plan(sessions=sessions, planned_km=total)


def subprocess_gate(payload: str, timeout: int = 35) -> dict:
    """An unavailable/crashed validator is a denial, never an implied success."""
    try:
        res = subprocess.run([sys.executable, '-m', 'merlino.gate_worker'],
                             input=json.dumps({'candidate_ttl': payload}), text=True,
                             capture_output=True, timeout=timeout, cwd=str(ROOT))
        reply = json.loads(res.stdout)
        if res.returncode != 0 or not isinstance(reply.get('conforms'), bool):
            raise RuntimeError('Validation worker did not finish successfully: ' +
                               '; '.join(reply.get('violations', [])))
        return reply
    except Exception as exc:
        return {'conforms': False, 'violations': ['GATE_UNAVAILABLE: ' + str(exc)],
                'warnings': [], 'gate': 'unavailable'}


class Merlino:
    def __init__(self, gate: Callable[[str], dict] | None = None):
        self.map = Graph().parse(ROOT / 'ontology' / 'merlin_map.ttl', format='turtle')
        self.railroad = Graph().parse(ROOT / 'reference' / 'bea' / 'teaching_aooe.ttl', format='turtle')
        self.railroad.parse(ROOT / 'ontology' / 'merlin_railroad.ttl', format='turtle')
        self.policy = Graph().parse(ROOT / 'ontology' / 'policy.ttl', format='turtle')
        self._gate = gate or subprocess_gate
        self.state = MR.Start
        self.baseline_km = Decimal('33.0')
        self.history_age_days = 2
        self.history_present = False
        self.pain_reported = False
        self.plan: Plan | None = None
        self.approval: Approval | None = None
        self.started = False
        self.trace: list[Decision] = []
        self.provenance = Graph()
        self.provenance.bind('prov', PROV)
        self.provenance.bind('mg', MG)

    @property
    def policy_version(self) -> str:
        return str(self.policy.value(MG.coachingPolicy, MG.version))

    @property
    def meaning_revision(self) -> str:
        return graph_hash(self.map)

    @property
    def max_ratio(self) -> Decimal:
        return Decimal(str(self.policy.value(MG.coachingPolicy, MG.maxWeeklyIncrease)))

    @property
    def max_age_days(self) -> int:
        return int(self.policy.value(MG.coachingPolicy, MG.maxHistoryAgeDays))

    @property
    def state_name(self) -> str:
        return str(self.state).split('#')[-1]

    def context(self) -> dict:
        """Only synthetic relevant facts; no raw notebook, user conversation or keys."""
        return {'runner': 'Alex (fictional)', 'baseline_week_km': float(self.baseline_km),
                'history_age_days': self.history_age_days if self.history_present else None,
                'history_read': self.history_present, 'state': self.state_name,
                'policy_version': self.policy_version, 'max_weekly_ratio': float(self.max_ratio),
                'pain_reported': self.pain_reported,
                'tempo_hard': bool(self.map.value(MR.TempoRun, MR.isHard)),
                'current_plan_km': float(self.plan.planned_km) if self.plan else None,
                'approved': self.approval is not None}

    def _transition(self, action: URIRef, actor: URIRef) -> URIRef | None:
        if (action, OPS.allowedRole, actor) not in self.railroad:
            return None
        for t in self.railroad.subjects(OPS.fromState, self.state):
            if (t, OPS.viaAction, action) in self.railroad:
                return self.railroad.value(t, OPS.toState)
        return None

    def _candidate(self, action: URIRef, actor: URIRef, plan: Plan | None = None) -> str:
        g = Graph()
        g.bind('mg', MG); g.bind('mr', MR); g.bind('ops', OPS)
        g += self.map; g += self.railroad; g += self.policy
        c = MG.currentCandidate
        g.add((c, RDF.type, MG.Candidate))
        for p, val in ((MG.action, action), (MG.actor, actor), (MG.fromState, self.state),
                       (MG.hasPain, Literal(self.pain_reported, datatype=XSD.boolean)),
                       (MG.historyPresent, Literal(self.history_present, datatype=XSD.boolean)),
                       (MG.historyAgeDays, Literal(self.history_age_days, datatype=XSD.integer)),
                       (MG.baselineKm, Literal(self.baseline_km, datatype=XSD.decimal)),
                       (MG.policyVersion, Literal(self.policy_version)),
                       (MG.meaningRevision, Literal(self.meaning_revision))):
            g.add((c, p, val))
        if plan is not None:
            g.add((c, MG.plannedKm, Literal(plan.planned_km, datatype=XSD.decimal)))
            g.add((c, MG.planDigest, Literal(plan.digest())))
            for i, session in enumerate(plan.sessions):
                node = MG[f'candidateSession{i+1}']
                g.add((c, MG.session, node))
                g.add((node, MG.dayNumber, Literal(session.day, datatype=XSD.integer)))
                g.add((node, MG.distanceKm, Literal(session.km, datatype=XSD.decimal)))
                g.add((node, MG.workoutType, session.kind))
        if self.approval is not None:
            g.add((c, MG.humanApproved, Literal(True, datatype=XSD.boolean)))
            g.add((c, MG.approvedDigest, Literal(self.approval.digest)))
            g.add((c, MG.approvedPolicyVersion, Literal(self.approval.policy_version)))
            g.add((c, MG.approvedMeaningRevision, Literal(self.approval.meaning_revision)))
        return g.serialize(format='turtle')

    def _precheck(self, action: URIRef, actor: URIRef, plan: Plan | None) -> list[str]:
        errors = []
        if self._transition(action, actor) is None:
            errors.append('RAILROAD: action/actor not permitted in current state')
        if self.pain_reported and action != MR.HandToHuman:
            errors.append('B3: pain already reported; human handover required')
        if action in (MR.ProposePlan, MR.StartWeek):
            if not self.history_present or self.history_age_days > self.max_age_days:
                errors.append('B1: history must be present and fresh')
            if plan is None:
                errors.append('B6: plan required')
            else:
                if len(plan.sessions) != 7 or len({s.day for s in plan.sessions}) != 7:
                    errors.append('B6: exactly seven distinct days required')
                if any(s.km < 0 or s.day < 1 or s.day > 7 or self.map.value(s.kind, MR.isHard) is None for s in plan.sessions):
                    errors.append('B6: invalid distance/day/workout type')
                if sum((s.km for s in plan.sessions), Decimal(0)) != plan.planned_km:
                    errors.append('B6: session sum must match declared total')
                if plan.planned_km > self.baseline_km * self.max_ratio:
                    errors.append('B2: plan exceeds versioned progression policy')
        if action == MR.StartWeek:
            if not self.approval or not plan or (self.approval.digest != plan.digest()
                    or self.approval.policy_version != self.policy_version
                    or self.approval.meaning_revision != self.meaning_revision):
                errors.append('B4: approval must match exact plan, policy and ontology revision')
        return errors

    def _dispatch(self, action: URIRef, actor: URIRef, plan: Plan | None = None) -> Decision:
        before = self.state_name
        ttl = self._candidate(action, actor, plan)
        local_errors = self._precheck(action, actor, plan)
        verdict = self._gate(ttl)
        gate_ok = verdict.get('conforms') is True
        errors = local_errors + list(verdict.get('violations', []))
        good = gate_ok and not errors
        target = self._transition(action, actor) if good else None
        good = good and target is not None
        dec = Decision(action=str(action).split('#')[-1], accepted=bool(good), committed=False,
                       from_state=before, to_state=str(target).split('#')[-1] if good else before,
                       violations=errors, warnings=list(verdict.get('warnings', [])),
                       gate=str(verdict.get('gate', '?')),
                       candidate_hash=hashlib.sha256(ttl.encode()).hexdigest()[:16])
        self.trace.append(dec)
        self._record_trace(dec)
        return dec

    def _record_trace(self, decision: Decision):
        node = TRACE[f'decision-{len(self.trace)}']
        self.provenance.add((node, RDF.type, PROV.Activity))
        self.provenance.add((node, MG.actionName, Literal(decision.action)))
        self.provenance.add((node, MG.accepted, Literal(decision.accepted)))
        self.provenance.add((node, MG.policyVersion, Literal(self.policy_version)))
        self.provenance.add((node, MG.meaningRevision, Literal(self.meaning_revision)))
        self.provenance.add((node, MG.candidateHash, Literal(decision.candidate_hash)))
        for error in decision.violations:
            self.provenance.add((node, MG.violation, Literal(error)))
        for warning in decision.warnings:
            self.provenance.add((node, MG.warning, Literal(warning)))

    def _commit(self, dec: Decision, new_state: URIRef, *, plan: Plan | None = None):
        if not dec.accepted:
            return
        self.state = new_state
        if plan is not None:
            self.plan = plan
            self.approval = None
        dec.committed = True
        node = TRACE[f'decision-{len(self.trace)}']
        self.provenance.add((node, MG.committed, Literal(True, datatype=XSD.boolean)))
        self.provenance.add((node, MG.toState, new_state))

    def read_history(self) -> Decision:
        dec = self._dispatch(MR.ReadRunHistory, MR.Merlin)
        if dec.accepted:
            self.history_present = True
            self._commit(dec, MR.BaselineKnown)
        return dec

    def propose(self, km: float, style: str = 'normal') -> Decision:
        try:
            plan = example_plan(km, style)
        except (ValueError, TypeError, OverflowError, InvalidOperation) as exc:
            return self._rejected('ProposePlan', 'INPUT: ' + str(exc))
        dec = self._dispatch(MR.ProposePlan, MR.Merlin, plan)
        if dec.accepted:
            self._commit(dec, MR.PlanProposed, plan=plan)
        return dec

    def approve_as_runner(self, *, human_confirmed: bool = False) -> Decision:
        """Out-of-band human action; NOT part of the model tool registry."""
        if not human_confirmed:
            return self._rejected('ApprovePlan', 'AUTH: explicit runner confirmation required')
        if self.plan is None:
            return self._rejected('ApprovePlan', 'AUTH: no validated plan')
        dec = self._dispatch(MR.ApprovePlan, MR.Runner, self.plan)
        if dec.accepted:
            self.approval = Approval(self.plan.digest(), self.policy_version, self.meaning_revision)
            self._commit(dec, MR.PlanApproved)
        return dec

    def start_week(self) -> Decision:
        if self.plan is None:
            return self._rejected('StartWeek', 'B6: plan missing')
        dec = self._dispatch(MR.StartWeek, MR.Merlin, self.plan)
        if dec.accepted:
            self._commit(dec, MR.WeekInProgress)
            self.started = True
        return dec

    def report_pain_as_runner(self, *, human_confirmed: bool = False) -> str:
        if not human_confirmed:
            raise PermissionError('Only a separately confirmed runner check-in may update pain evidence')
        self.pain_reported = True
        return 'Pain evidence registered in synthetic runner context; agent must hand over.'

    def handover(self) -> Decision:
        dec = self._dispatch(MR.HandToHuman, MR.Merlin)
        if dec.accepted:
            self._commit(dec, MR.HandedToHuman)
        return dec

    def revise_tempo_meaning(self, *, tempo_is_hard: bool, steward_confirmed: bool = False):
        """Facilitator-only ontology edit: outside the model tool surface."""
        if not steward_confirmed:
            raise PermissionError('A semantic steward must authorize this ontology edit')
        self.map.remove((MR.TempoRun, MR.isHard, None))
        self.map.add((MR.TempoRun, MR.isHard, Literal(tempo_is_hard, datatype=XSD.boolean)))
        self.approval = None
        return {'tempo_is_hard': tempo_is_hard, 'meaning_revision': self.meaning_revision,
                'approval_invalidated': True}

    def revise_policy(self, *, ratio: float, version: str, steward_confirmed: bool = False):
        if not steward_confirmed:
            raise PermissionError('A policy owner must authorize this edit')
        if not (Decimal('1.0') <= Decimal(str(ratio)) <= Decimal('1.5')):
            raise ValueError('Workshop ratio must be in [1.0, 1.5]')
        if not version.startswith('workshop-') or version == self.policy_version:
            raise ValueError('Use a new workshop-* policy version')
        self.policy.set((MG.coachingPolicy, MG.maxWeeklyIncrease, Literal(Decimal(str(ratio)), datatype=XSD.decimal)))
        self.policy.set((MG.coachingPolicy, MG.version, Literal(version)))
        self.approval = None
        return {'policy_version': version, 'ratio': ratio, 'approval_invalidated': True}

    def _rejected(self, action: str, message: str) -> Decision:
        dec = Decision(action, False, False, self.state_name, self.state_name,
                       [message], [], 'input/authorization boundary', '')
        self.trace.append(dec); self._record_trace(dec)
        return dec

    def trace_json(self) -> str:
        return json.dumps([d.as_dict() for d in self.trace], indent=2)

    def trace_turtle(self) -> str:
        return self.provenance.serialize(format='turtle')
