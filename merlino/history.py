"""Graph-derived synthetic runner evidence, not constants or user-provided model claims.

Date is injected for repeatability: Colab's current wall-clock date must not
silently convert this fixed 2026 workshop fixture into stale data.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

SCHEMA = Namespace('https://schema.org/')
MR = Namespace('https://example.org/merlin-railroad#')
EX = Namespace('https://example.org/kg/')
WORKOUTS = {'easy': MR.EasyRun, 'long': MR.LongRun, 'intervals': MR.Intervals,
            'tempo': MR.TempoRun, 'rest': MR.RestDay}


def run_rows(graph: Graph) -> list[tuple[date, Decimal, URIRef]]:
    """Return date, numeric km, workout IRI from graph; reject ambiguous measurements."""
    runs = []
    for subject in graph.subjects(RDF.type, SCHEMA.ExerciseAction):
        timestamp = graph.value(subject, SCHEMA.startTime)
        km = graph.value(subject, MR.distanceKm) or graph.value(subject, SCHEMA.distance)
        workout = graph.value(subject, MR.workoutType)
        if timestamp is None or km is None or workout is None:
            raise ValueError('Run lacks date, distance, or workout type')
        when = timestamp.toPython()
        day = when.date() if isinstance(when, datetime) else when
        distance = Decimal(str(km))
        if not distance.is_finite() or distance < 0:
            raise ValueError('Run distance must be finite and nonnegative')
        runs.append((day, distance, workout))
    if not runs:
        raise ValueError('Training map has no run history')
    return sorted(runs)


def history_summary(graph: Graph, today: date | None = None) -> dict:
    rows = run_rows(graph)
    latest = rows[-1][0]
    anchor = today or (latest + timedelta(days=2))
    age = (anchor - latest).days
    if age < 0:
        raise ValueError('Workshop clock cannot precede the latest run')
    lower = latest - timedelta(days=6)
    baseline = sum((km for day, km, _ in rows if lower <= day <= latest), Decimal(0))
    if baseline <= 0:
        raise ValueError('No positive baseline in the last seven run-history days')
    return {'last_run_date': latest, 'as_of_date': anchor, 'age_days': age,
            'baseline_km': baseline, 'runs': len(rows)}


def load_synthetic_csv(path: str | Path, *, name: str = 'Workshop runner') -> Graph:
    """CSV import for synthetic data only. Participants must not upload real runs to Colab."""
    graph = Graph()
    graph.add((EX['runner-csv'], RDF.type, SCHEMA.Person))
    graph.add((EX['runner-csv'], SCHEMA.name, Literal(name)))
    with open(path, newline='', encoding='utf-8') as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            day = date.fromisoformat(row['date'])
            workout = WORKOUTS[row['workout_type'].strip().lower()]
            km = Decimal(row['distance_km'])
            if not km.is_finite() or km < 0:
                raise ValueError('Invalid distance')
            subject = EX[f'csv-run-{i+1}']
            graph.add((subject, RDF.type, SCHEMA.ExerciseAction))
            graph.add((subject, SCHEMA.agent, EX['runner-csv']))
            graph.add((subject, SCHEMA.startTime, Literal(day.isoformat()+'T08:00:00Z', datatype=XSD.dateTime)))
            graph.add((subject, MR.distanceKm, Literal(km, datatype=XSD.decimal)))
            graph.add((subject, MR.workoutType, workout))
    return graph
