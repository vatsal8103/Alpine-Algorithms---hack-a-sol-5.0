"""Sample vehicle registry loading and lookup."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .plate import clean_text, is_valid_plate
from .schemas import RegistryEntry


class Registry:
    def __init__(self, entries: list[RegistryEntry], warnings: list[str] | None = None):
        self.entries = entries
        self.warnings = warnings or []
        self.by_plate = {e.plate: e for e in entries}

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, plate: str) -> RegistryEntry | None:
        return self.by_plate.get(clean_text(plate))

    def filter(
        self,
        body_type: str | None = None,
        color: str | None = None,
        state: str | None = None,
    ) -> list[RegistryEntry]:
        out = self.entries
        if body_type and body_type != "unknown":
            out = [e for e in out if e.body_type == body_type]
        if color and color != "unknown":
            out = [e for e in out if e.color == color]
        if state:
            out = [e for e in out if e.state == state]
        return out

    def summary(self) -> dict:
        colors: dict[str, int] = {}
        bodies: dict[str, int] = {}
        states: dict[str, int] = {}
        for e in self.entries:
            colors[e.color] = colors.get(e.color, 0) + 1
            bodies[e.body_type] = bodies.get(e.body_type, 0) + 1
            states[e.state] = states.get(e.state, 0) + 1
        return {
            "count": len(self.entries),
            "colors": dict(sorted(colors.items(), key=lambda kv: -kv[1])),
            "body_types": dict(sorted(bodies.items(), key=lambda kv: -kv[1])),
            "states": dict(sorted(states.items(), key=lambda kv: -kv[1])),
        }


def _entry_from_dict(row: dict) -> RegistryEntry | None:
    plate = clean_text(row.get("plate") or row.get("registration_number") or "")
    if not plate or not is_valid_plate(plate):
        return None
    state = (row.get("state") or plate[:2]).strip().upper()
    try:
        year = int(float(row.get("year") or 0))
    except (TypeError, ValueError):
        year = 0
    body_type = (row.get("body_type") or "unknown").strip().lower()
    fuel = (row.get("fuel") or "unknown").strip().lower()
    category = (row.get("category") or "").strip().lower()
    if category not in ("private", "commercial", "ev"):
        if fuel in ("electric", "ev"):
            category = "ev"
        elif body_type in ("truck", "bus", "auto-rickshaw"):
            category = "commercial"
        else:
            category = "private"
    plate_style = (row.get("plate_style") or "").strip().lower()
    if plate_style not in ("white", "yellow", "green"):
        plate_style = {"ev": "green", "commercial": "yellow"}.get(category, "white")
    return RegistryEntry(
        vehicle_id=str(row.get("vehicle_id") or plate),
        plate=plate,
        make=(row.get("make") or "unknown").strip(),
        model=(row.get("model") or "unknown").strip(),
        color=(row.get("color") or row.get("colour") or "unknown").strip().lower(),
        body_type=body_type,
        plate_style=plate_style,
        category=category,
        state=state,
        fuel=fuel,
        year=year,
        owner_city=(row.get("owner_city") or "").strip(),
    )


def load_registry(path: str | Path) -> Registry:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    warnings: list[str] = []
    rows: list[dict] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("vehicles", [])
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    entries: list[RegistryEntry] = []
    seen: set[str] = set()
    for row in rows:
        entry = _entry_from_dict(row)
        if entry is None:
            warnings.append(f"Skipped invalid registry row: {row}")
            continue
        if entry.plate in seen:
            warnings.append(f"Duplicate plate skipped: {entry.plate}")
            continue
        seen.add(entry.plate)
        entries.append(entry)
    if not entries:
        raise ValueError(f"Registry contains no valid entries: {path}")
    return Registry(entries, warnings)


def save_registry(entries: list[RegistryEntry], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "vehicle_id", "plate", "make", "model", "color", "body_type",
        "plate_style", "category", "state", "fuel", "year", "owner_city",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for e in entries:
            writer.writerow(e.model_dump())
