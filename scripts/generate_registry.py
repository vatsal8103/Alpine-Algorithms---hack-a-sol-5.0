"""Generate a deterministic sample Indian vehicle registry.

Run:
    python scripts/generate_registry.py --count 140
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracex.plate import is_valid_plate
from tracex.registry import save_registry
from tracex.schemas import RegistryEntry

RTO_POOL = [
    ("JH", 5), ("JH", 1), ("MH", 12), ("MH", 1), ("KA", 3), ("KA", 1),
    ("CG", 4), ("CG", 1), ("UP", 32), ("UP", 16), ("DL", 8), ("DL", 12),
    ("TN", 9), ("TN", 1), ("MP", 9), ("RJ", 14), ("GJ", 1), ("WB", 2),
    ("TS", 9), ("AP", 16), ("KL", 7), ("HR", 26), ("PB", 10), ("OD", 2),
    ("BR", 1), ("AS", 1), ("UK", 7),
]

CITIES = {
    "JH": "Ranchi", "MH": "Pune", "KA": "Bengaluru", "CG": "Raipur",
    "UP": "Lucknow", "DL": "New Delhi", "TN": "Chennai", "MP": "Indore",
    "RJ": "Jaipur", "GJ": "Ahmedabad", "WB": "Kolkata", "TS": "Hyderabad",
    "AP": "Vijayawada", "KL": "Kochi", "HR": "Gurugram", "PB": "Ludhiana",
    "OD": "Bhubaneswar", "BR": "Patna", "AS": "Guwahati", "UK": "Dehradun",
}

MODELS = {
    "hatchback": [
        ("Maruti Suzuki", "Swift"), ("Maruti Suzuki", "Baleno"), ("Hyundai", "i20"),
        ("Tata", "Altroz"), ("Honda", "Jazz"), ("Renault", "Kwid"),
    ],
    "sedan": [
        ("Honda", "City"), ("Hyundai", "Verna"), ("Maruti Suzuki", "Ciaz"),
        ("Skoda", "Slavia"), ("Volkswagen", "Virtus"),
    ],
    "suv": [
        ("Mahindra", "Scorpio"), ("Tata", "Nexon"), ("Hyundai", "Creta"),
        ("Kia", "Seltos"), ("Toyota", "Fortuner"), ("Maruti Suzuki", "Brezza"),
    ],
    "two-wheeler": [
        ("Hero", "Splendor"), ("Bajaj", "Pulsar"), ("Honda", "Activa"),
        ("TVS", "Apache"), ("Royal Enfield", "Classic 350"), ("Yamaha", "FZ"),
    ],
    "auto-rickshaw": [
        ("Bajaj", "RE"), ("Piaggio", "Ape"), ("Mahindra", "Alfa"),
    ],
    "truck": [
        ("Tata", "407"), ("Ashok Leyland", "Dost"), ("Eicher", "Pro 2049"),
    ],
    "bus": [
        ("Tata", "Starbus"), ("Ashok Leyland", "Viking"),
    ],
}

COLORS = ["white", "silver", "black", "red", "blue", "grey", "green", "yellow", "orange", "brown"]
COLOR_WEIGHTS = [30, 18, 18, 10, 8, 5, 4, 3, 2, 2]

DEMO_ENTRIES = [
    RegistryEntry(
        vehicle_id="demo-jh05-suv", plate="JH05AB6799", make="Mahindra", model="Scorpio",
        color="white", body_type="suv", plate_style="white", category="private",
        state="JH", fuel="diesel", year=2021, owner_city="Ranchi",
    ),
    RegistryEntry(
        vehicle_id="demo-mh12-hatch", plate="MH12DE1433", make="Maruti Suzuki", model="Swift",
        color="red", body_type="hatchback", plate_style="white", category="private",
        state="MH", fuel="petrol", year=2019, owner_city="Pune",
    ),
    RegistryEntry(
        vehicle_id="demo-ka03-bike", plate="KA03MN4821", make="Royal Enfield", model="Classic 350",
        color="black", body_type="two-wheeler", plate_style="white", category="private",
        state="KA", fuel="petrol", year=2020, owner_city="Bengaluru",
    ),
    RegistryEntry(
        vehicle_id="demo-cg04-auto", plate="CG04TR7788", make="Bajaj", model="RE",
        color="yellow", body_type="auto-rickshaw", plate_style="yellow", category="commercial",
        state="CG", fuel="cng", year=2018, owner_city="Raipur",
    ),
    RegistryEntry(
        vehicle_id="demo-up32-truck", plate="UP32KL9021", make="Tata", model="407",
        color="blue", body_type="truck", plate_style="yellow", category="commercial",
        state="UP", fuel="diesel", year=2017, owner_city="Lucknow",
    ),
    RegistryEntry(
        vehicle_id="demo-dl8-ev", plate="DL8CAF1234", make="Tata", model="Nexon EV",
        color="green", body_type="suv", plate_style="green", category="ev",
        state="DL", fuel="electric", year=2023, owner_city="New Delhi",
    ),
]


def make_entry(plate: str, rng: random.Random) -> RegistryEntry:
    state = plate[:2]
    body = rng.choices(
        ["hatchback", "sedan", "suv", "two-wheeler", "auto-rickshaw", "truck", "bus"],
        weights=[28, 14, 22, 20, 7, 6, 3],
    )[0]
    make, model = rng.choice(MODELS[body])
    color = rng.choices(COLORS, weights=COLOR_WEIGHTS)[0]
    category = "commercial" if body in ("auto-rickshaw", "truck", "bus") else "private"
    plate_style = "yellow" if category == "commercial" else "white"
    fuel = rng.choice(["petrol", "diesel", "cng"]) if body != "two-wheeler" else "petrol"
    return RegistryEntry(
        vehicle_id=f"veh-{plate.lower()}",
        plate=plate,
        make=make,
        model=model,
        color=color,
        body_type=body,
        plate_style=plate_style,
        category=category,
        state=state,
        fuel=fuel,
        year=rng.randint(2012, 2024),
        owner_city=CITIES.get(state, "Unknown"),
    )


def generate(count: int = 140, seed: int = 42) -> list[RegistryEntry]:
    rng = random.Random(seed)
    entries = list(DEMO_ENTRIES)
    seen = {e.plate for e in entries}
    attempts = 0
    while len(entries) < count and attempts < count * 40:
        attempts += 1
        state, rto = rng.choice(RTO_POOL)
        series = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
        number = rng.randint(1000, 9999)
        if rto < 10 and rng.random() < 0.25:
            series = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))
            plate = f"{state}{rto}{series}{number:04d}"
        else:
            plate = f"{state}{rto:02d}{series}{number:04d}"
        if plate in seen or not is_valid_plate(plate):
            continue
        seen.add(plate)
        entries.append(make_entry(plate, rng))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=140)
    parser.add_argument("--out", default="data/registry/vehicles.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    entries = generate(args.count, args.seed)
    save_registry(entries, args.out)
    print(f"Wrote {len(entries)} vehicles to {args.out}")


if __name__ == "__main__":
    main()
