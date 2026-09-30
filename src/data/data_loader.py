"""Loads activities.csv into plain Python dicts, parsing the JSON-encoded
list fields (interests, purpose, instructions, tags) back into lists.

This is the single place that knows the CSV's on-disk encoding, so
ingestion.py and anything else that needs activities never touches csv/json
directly.
"""
import csv
import json
from typing import Any

from src.config import ACTIVITIES_CSV

LIST_FIELDS = ("interests", "purpose", "instructions", "tags")
INT_FIELDS = ("duration_minutes",)


def load_activities(csv_path: str = ACTIVITIES_CSV) -> list[dict[str, Any]]:
    """Read activities.csv and return a list of activity dicts with list
    fields parsed and duration_minutes as int."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No activities found in {csv_path}")

    activities = []
    for row in rows:
        activity = dict(row)
        for field in LIST_FIELDS:
            activity[field] = json.loads(activity[field])
        for field in INT_FIELDS:
            activity[field] = int(activity[field])
        activities.append(activity)
    return activities
