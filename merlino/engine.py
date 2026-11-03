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
from datetime import date, timedelta, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable
from .history import history_summary, SCHEMA

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, PROV, XSD, RDFS, SKOS

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
    def __init__(self, gate: Callable[[str], dict] | None = None, *,
                 runner_map: Graph | None = None, today: date | None = None):
        self.map = Graph().parse(ROOT / 'ontology' / 'merlin_map.ttl', format='turtle')
        if runner_map is not None:
            # Only replace the fictional observations; keep the shared domain
            # ontology so the same semantics apply to uploaded synthetic CSVs.
            for run in list(self.map.subjects(RDF.type, SCHEMA.ExerciseAction)):
                self.map.remove((run, None, None))
            self.map += runner_map
        self.railroad = Graph().parse(ROOT / 'reference' / 'bea' / 'teaching_aooe.ttl', format='turtle')
        self.railroad.parse(ROOT / 'ontology' / 'merlin_railroad.ttl', format='turtle')
        self.policy = Graph().parse(ROOT / 'ontology' / 'policy.ttl', format='turtle')
        self._gate = gate or subprocess_gate
        self.state = MR.Start
        source = history_summary(self.map, today)
        self.baseline_km = source['baseline_km']
        self.history_age_days = source['age_days']
        self.today = source['as_of_date']
        self.source_history = source
        self.history_present = False
        self.pain_reported = False
        self.latest_checkin: dict | None = None
        self.checkins: list[dict] = []
        self.logged_runs: list[dict] = []
        self.evidence: set[URIRef] = set()
        self.checked_proposals: set[tuple[str, str, str]] = set()
        self.started_plans: list[Plan] = []
        self.handed_over_reason: str | None = None
        self.plan: Plan | None = None
        self.approval: Approval | None = None
        self.started = False
        self.trace: list[Decision] = []
        self.provenance = Graph()
        self.memory_trace = Graph()  # prior RDF episodes imported by the facilitator only
        self.provenance.bind('prov', PROV)
        self.provenance.bind('mg', MG)

    @property
    def policy_version(self) -> str:
        return str(self.policy.value(MG.coachingPolicy, MG.version))

    @property
    def meaning_revision(self) -> str:
        # Only the ontology's meaning-bearing classes/properties. Runs are data, not a schema revision.
        terms = Graph()
        for subject in self.map.subjects(RDF.type, SKOS.Concept):
            for predicate, value in self.map.predicate_objects(subject):
                terms.add((subject, predicate, value))
        return graph_hash(terms)

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
                'last_run_date': self.source_history['last_run_date'].isoformat(),
                'reference_date': self.today.isoformat(),
                'logged_runs': len(self.logged_runs), 'latest_check_in': self.latest_checkin,
                'policy_version': self.policy_version, 'max_weekly_ratio': float(self.max_ratio),
                'pain_reported': self.pain_reported,
                'tempo_hard': bool(self.map.value(MR.TempoRun, MR.isHard)),
                'current_plan_km': float(self.plan.planned_km) if self.plan else None,
                'approved': self.approval is not None,
                'remembered_decisions': self.recall_memory(limit=3)}

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
        # Runner-context facts are authoritative host state, not model-selected
        # evidence. An older pain-free record cannot replace the latest check-in.
        context = MG.runnerContext
        g.add((context, MG.activePainHold, Literal(self.pain_reported, datatype=XSD.boolean)))
        if self.latest_checkin is not None:
            latest = MG['latestCheckIn-' + str(len(self.checkins))]
            g.add((context, MG.latestCheckIn, latest))
            g.add((latest, RDF.type, MR.CheckIn))
            g.add((latest, MG.hasPain, Literal(self.latest_checkin['pain'], datatype=XSD.boolean)))
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
        if plan is not None and (plan.digest(), self.policy_version, self.meaning_revision) in self.checked_proposals:
            g.add((c, MG.wasProposed, Literal(True, datatype=XSD.boolean)))
        for kind in self.evidence:
            node = MG['evidence-' + str(kind).split('#')[-1]]
            g.add((c, MG.hasEvidence, node))
            g.add((node, RDF.type, kind))
        if self.approval is not None:
            g.add((c, MG.humanApproved, Literal(True, datatype=XSD.boolean)))
            g.add((c, MG.approvedDigest, Literal(self.approval.digest)))
            g.add((c, MG.approvedPolicyVersion, Literal(self.approval.policy_version)))
            g.add((c, MG.approvedMeaningRevision, Literal(self.approval.meaning_revision)))
        return g.serialize(format='turtle')

    def _precheck(self, action: URIRef, actor: URIRef, plan: Plan | None) -> list[str]:
        errors = []
        if not any((transition, OPS.viaAction, action) in self.railroad
                   for transition in self.railroad.subjects(OPS.fromState, self.state)):
            errors.append('G2: action is off the railroad in this state')
        if (action, OPS.allowedRole, actor) not in self.railroad:
            errors.append('G3: actor role is not permitted')
        for required in self.railroad.objects(action, OPS.requiresEvidence):
            if required not in self.evidence:
                errors.append('G4: missing required evidence ' + str(required).split('#')[-1])
        if self.pain_reported and action != MR.HandToHuman and (
                actor == MR.Merlin or action in (MR.ApprovePlan, MR.StartWeek)):
            errors.append('B3: pain reported; agent actions and training authorization are blocked')
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
                if sum((session.km for session in plan.sessions), Decimal(0)) > self.baseline_km * self.max_ratio:
                    errors.append('B2: session-summed distance exceeds versioned progression policy')
        if action in (MR.ApprovePlan, MR.StartWeek) and (plan is None or (plan.digest(), self.policy_version, self.meaning_revision) not in self.checked_proposals):
            errors.append('B7: only a checked proposal can be approved or started')
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
        observation = TRACE[f'observation-{len(self.trace)}']
        validation = TRACE[f'validation-{len(self.trace)}']
        self.provenance.add((observation, RDF.type, PROV.Entity))
        self.provenance.add((observation, MG.observedState, Literal(decision.from_state)))
        self.provenance.add((node, RDF.type, PROV.Activity))
        self.provenance.add((node, PROV.used, observation))
        self.provenance.add((validation, RDF.type, PROV.Activity))
        self.provenance.add((validation, PROV.wasInformedBy, node))
        self.provenance.add((validation, MG.conforms, Literal(decision.accepted, datatype=XSD.boolean)))
        self.provenance.add((node, MG.actionName, Literal(decision.action)))
        self.provenance.add((node, MG.accepted, Literal(decision.accepted)))
        self.provenance.add((node, MG.policyVersion, Literal(self.policy_version)))
        self.provenance.add((node, MG.meaningRevision, Literal(self.meaning_revision)))
        self.provenance.add((node, MG.candidateHash, Literal(decision.candidate_hash)))
        for error in decision.violations:
            self.provenance.add((validation, MG.violation, Literal(error)))
            self.provenance.add((node, MG.violation, Literal(error)))
        for warning in decision.warnings:
            self.provenance.add((validation, MG.warning, Literal(warning)))
            self.provenance.add((node, MG.warning, Literal(warning)))

    def _commit(self, dec: Decision, new_state: URIRef, *, plan: Plan | None = None):
        if not dec.accepted:
            return
        self.state = new_state
        if plan is not None:
            self.plan = plan
            self.approval = None
            self.evidence.discard(MR.PlanApproval)
        dec.committed = True
        node = TRACE[f'decision-{len(self.trace)}']
        self.provenance.add((node, MG.committed, Literal(True, datatype=XSD.boolean)))
        self.provenance.add((node, MG.toState, new_state))
        execution = TRACE[f'execution-{len(self.trace)}']
        self.provenance.add((execution, RDF.type, PROV.Activity))
        self.provenance.add((execution, PROV.wasInformedBy, TRACE[f'validation-{len(self.trace)}']))
        self.provenance.add((execution, MG.actionName, Literal(dec.action)))

    def read_history(self) -> Decision:
        dec = self._dispatch(MR.ReadRunHistory, MR.Merlin)
        if dec.accepted:
            self.history_present = True
            self.evidence.add(MR.RunHistory)
            self._commit(dec, MR.BaselineKnown)
        return dec

    def propose(self, km: float, style: str = 'normal') -> Decision:
        try:
            plan = example_plan(km, style)
        except (ValueError, TypeError, OverflowError, InvalidOperation) as exc:
            return self._rejected('ProposePlan', 'INPUT: ' + str(exc))
        dec = self._dispatch(MR.ProposePlan, MR.Merlin, plan)
        if dec.accepted:
            self.checked_proposals.add((plan.digest(), self.policy_version, self.meaning_revision))
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
            self.evidence.add(MR.PlanApproval)
            self._commit(dec, MR.PlanApproved)
        return dec

    def start_week(self) -> Decision:
        if self.plan is None:
            return self._rejected('StartWeek', 'B6: plan missing')
        dec = self._dispatch(MR.StartWeek, MR.Merlin, self.plan)
        if dec.accepted:
            self._commit(dec, MR.WeekInProgress)
            self.started = True
            self.started_plans.append(self.plan)
        return dec

    def report_pain_as_runner(self, *, human_confirmed: bool = False) -> str:
        if not human_confirmed:
            raise PermissionError('Only a separately confirmed runner check-in may update pain evidence')
        self.pain_reported = True
        self.latest_checkin = {'pain': True, 'runner_confirmed': True}
        self.checkins.append(self.latest_checkin)
        self.evidence.add(MR.CheckIn)
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
        self.evidence.discard(MR.PlanApproval)
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
        self.evidence.discard(MR.PlanApproval)
        return {'policy_version': version, 'ratio': ratio, 'approval_invalidated': True}

    def log_run_as_runner(self, km: float, workout: str = 'easy', *, human_confirmed: bool = False) -> Decision:
        """Runner-authorized observation; append to map only AFTER independent gate validation."""
        if not human_confirmed:
            return self._rejected('LogRun', 'AUTH: runner confirmation required')
        kinds = {'easy': MR.EasyRun, 'long': MR.LongRun, 'intervals': MR.Intervals,
                 'tempo': MR.TempoRun}
        try:
            distance = Decimal(str(km))
            if not distance.is_finite() or not Decimal(0) < distance <= Decimal(42):
                raise ValueError('invalid synthetic logged distance')
            kind = kinds[workout]
        except (ValueError, KeyError, InvalidOperation, TypeError):
            return self._rejected('LogRun', 'INPUT: workout type or distance invalid')
        dec = self._dispatch(MR.LogRun, MR.Runner)
        if dec.accepted:
            self._commit(dec, self._transition(MR.LogRun, MR.Runner))
            self.today += timedelta(days=1)
            record = EX[f'logged-run-{len(self.logged_runs)+1}']
            self.map.add((record, RDF.type, SCHEMA.ExerciseAction))
            self.map.add((record, SCHEMA.startTime,
                          Literal(self.today.isoformat()+'T07:00:00Z', datatype=XSD.dateTime)))
            self.map.add((record, MR.distanceKm, Literal(distance, datatype=XSD.decimal)))
            self.map.add((record, MR.workoutType, kind))
            self.logged_runs.append({'km': float(distance), 'workout': workout, 'day': self.today.isoformat()})
        return dec

    def ask_check_in(self) -> Decision:
        dec = self._dispatch(MR.AskCheckIn, MR.Merlin)
        if dec.accepted:
            self._commit(dec, MR.AwaitingCheckIn)
        return dec

    def answer_check_in_as_runner(self, *, has_pain: bool, effort: int = 5,
                                  sleep: int = 4, human_confirmed: bool = False) -> Decision:
        if not human_confirmed:
            return self._rejected('AnswerCheckIn', 'AUTH: runner confirmation required')
        if type(has_pain) is not bool or type(effort) is not int or not 1 <= effort <= 10 or \
                type(sleep) is not int or not 1 <= sleep <= 5:
            return self._rejected('AnswerCheckIn', 'INPUT: invalid synthetic check-in')
        dec = self._dispatch(MR.AnswerCheckIn, MR.Runner)
        if dec.accepted:
            self._commit(dec, MR.WeekInProgress)
            record = {'pain': has_pain, 'effort': effort, 'sleep': sleep,
                      'runner_confirmed': True}
            self.latest_checkin = record
            self.checkins.append(record)
            self.pain_reported = self.pain_reported or has_pain
            self.evidence.add(MR.CheckIn)
        return dec

    def review_week(self) -> Decision:
        dec = self._dispatch(MR.ReviewWeek, MR.Merlin)
        if dec.accepted:
            self._commit(dec, MR.WeekReviewed)
        return dec

    def close_week_as_runner(self, *, human_confirmed: bool = False) -> Decision:
        if not human_confirmed:
            return self._rejected('CloseWeek', 'AUTH: runner confirmation required')
        dec = self._dispatch(MR.CloseWeek, MR.Runner)
        if dec.accepted:
            self._commit(dec, MR.WeekClosed)
        return dec

    def _rejected(self, action: str, message: str) -> Decision:
        dec = Decision(action, False, False, self.state_name, self.state_name,
                       [message], [], 'input/authorization boundary', '')
        self.trace.append(dec); self._record_trace(dec)
        return dec

    def save_memory(self, path):
        """Export the synthetic RDF world, policy and audit to a local TriG file.

        A facilitator-only operation, never a model tool or remote write.
        """
        from .memory import save_memory
        return save_memory(self, path)

    def load_memory(self, path, *, facilitator_confirmed=False):
        """Restore the semantic world and prior RDF episodes, NOT runtime authority.

        A new session starts at Start; an old approval is NEVER restored.
        """
        if not facilitator_confirmed:
            raise PermissionError('Facilitator approval required for local memory import')
        from .memory import load_memory
        return load_memory(self, path)

    def recall_memory(self, limit=5):
        """Read governed past decisions from the loaded RDF provenance graph."""
        from .memory import recall_memory
        return recall_memory(self.memory_trace, limit=limit)

    def trace_json(self) -> str:
        return json.dumps([d.as_dict() for d in self.trace], indent=2)

    def trace_turtle(self) -> str:
        return self.provenance.serialize(format='turtle')
