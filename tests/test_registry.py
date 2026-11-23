"""Tests for synthetic registry generation and ground truth validation."""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
import pytest

from scripts.make_registry import (
    generate_registry,
    is_valid_plate,
    CONFUSABLE_MAP
)

INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")


def test_plate_format_validation():
    assert is_valid_plate("JH05AB6799")
    assert is_valid_plate("MH12DE1433")
    assert is_valid_plate("DL08CA1234")
    assert is_valid_plate("KA03MN4821")
    assert is_valid_plate("UP32KL9021")
    assert is_valid_plate("CG04TR7788")

    # Invalid plates: 0 series letters, 4 series letters, digits in state code, 3 digits, 5 digits
    assert not is_valid_plate("JH051234")      # Missing series letters
    assert not is_valid_plate("JH05ABCD6799")  # 4 series letters (max is 3)
    assert not is_valid_plate("1234AB5678")    # Digits in state code
    assert not is_valid_plate("JH05AB679")     # Only 3 trailing digits
    assert not is_valid_plate("JH05AB67999")   # 5 trailing digits


def test_registry_size_and_no_duplicates():
    reg, oor = generate_registry(total_count=500, seed=42, num_out_of_registry=25)
    assert len(reg) == 500
    assert len(oor) == 25

    plates = [r["registration_number"] for r in reg]
    unique_plates = set(plates)
    assert len(plates) == len(unique_plates), "Registry contains duplicate registration numbers!"

    for p in plates:
        assert is_valid_plate(p), f"Invalid Indian plate format: {p}"


def test_out_of_registry_disjoint():
    reg, oor = generate_registry(total_count=500, seed=42, num_out_of_registry=25)
    reg_plates = set(r["registration_number"] for r in reg)
    oor_plates = set(r["registration_number"] for r in oor)

    overlap = reg_plates.intersection(oor_plates)
    assert len(overlap) == 0, f"Out-of-registry targets overlap with registry: {overlap}"


def test_reproducibility():
    reg1, oor1 = generate_registry(total_count=100, seed=123, num_out_of_registry=10)
    reg2, oor2 = generate_registry(total_count=100, seed=123, num_out_of_registry=10)

    assert reg1 == reg2, "Registry generation is not deterministic for identical seeds!"
    assert oor1 == oor2, "Out-of-registry generation is not deterministic for identical seeds!"


def test_hard_negatives_exist():
    reg, _ = generate_registry(total_count=500, seed=42)
    plates = [r["registration_number"] for r in reg]
    plates_set = set(plates)

    # 1. Check for plates differing by exactly 1 character
    one_char_diff_found = False
    for i, p1 in enumerate(plates[:50]):
        for p2 in plates[i + 1:100]:
            if len(p1) == len(p2):
                diff = sum(1 for a, b in zip(p1, p2) if a != b)
                if diff == 1:
                    one_char_diff_found = True
                    break
        if one_char_diff_found:
            break
    assert one_char_diff_found, "Registry must contain hard negative plates differing by exactly 1 character!"

    # 2. Check for same colour and body type with different plates
    same_attr_found = False
    for i, r1 in enumerate(reg[:30]):
        for r2 in reg[i + 1:60]:
            if r1["colour"] == r2["colour"] and r1["body_type"] == r2["body_type"] and r1["registration_number"] != r2["registration_number"]:
                same_attr_found = True
                break
        if same_attr_found:
            break
    assert same_attr_found, "Registry must contain vehicles with same colour & body type but different plates!"


def test_sqlite_and_csv_generation(tmp_path):
    csv_file = tmp_path / "vehicles.csv"
    sqlite_file = tmp_path / "vehicles.sqlite"
    oor_file = tmp_path / "out_of_registry.csv"

    from scripts.make_registry import save_registry_files
    reg, oor = generate_registry(total_count=100, seed=42, num_out_of_registry=10)
    save_registry_files(reg, oor, csv_path=str(csv_file), db_path=str(sqlite_file), out_csv_path=str(oor_file))

    assert csv_file.exists()
    assert sqlite_file.exists()
    assert oor_file.exists()

    # Verify SQLite schema and count
    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM vehicles")
    count = cur.fetchone()[0]
    assert count == 100

    cur.execute("SELECT vehicle_id, registration_number, colour FROM vehicles LIMIT 1")
    row = cur.fetchone()
    assert len(row) == 3
    conn.close()
