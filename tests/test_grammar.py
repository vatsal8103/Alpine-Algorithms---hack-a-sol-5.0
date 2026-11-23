"""Table-driven test suite for Indian License Plate Grammar Parser and CMVR Constraints."""

import pytest
from fusion.grammar import (
    parse_plate,
    parse_plate_text,
    validate_plate_grammar,
    CURRENT_STATES,
    LEGACY_STATES,
)


@pytest.mark.parametrize("plate_str, expected_template", [
    ("MH12AB1234", "STANDARD"),
    ("MP04ZK6184", "STANDARD"),
    ("DL8CAF1234", "DELHI"),
    ("KA03MN4821", "STANDARD"),
    ("CG04TR7788", "STANDARD"),
    ("JH05AB6799", "STANDARD"),
    ("TG09AB1234", "STANDARD"),
    ("22BH1234AA", "BHARAT"),
    ("MH12A1234", "STANDARD"),
])
def test_valid_plates_table(plate_str: str, expected_template: str):
    """Test all verified valid Indian plate formats across templates."""
    parsed = parse_plate_text(plate_str)
    assert parsed.is_valid is True, f"Plate {plate_str} should be valid"
    assert parsed.template == expected_template
    assert parsed.text == plate_str
    assert "?" not in parsed.text
    assert validate_plate_grammar(plate_str) is True


def test_district_o_to_0_only_with_support():
    """'MPO4ZK6184': O in district slot becomes 0 ONLY with supporting alternatives, otherwise '?'."""
    # Case A: Pos 2 is 'O' with alternative evidence for '0'
    posteriors_with_support = [
        {"M": 0.95}, {"P": 0.95},
        {"O": 0.70, "0": 0.30},  # '0' supported by OCR posterior
        {"4": 0.95}, {"Z": 0.95}, {"K": 0.95},
        {"6": 0.95}, {"1": 0.95}, {"8": 0.95}, {"4": 0.95}
    ]
    parsed_supported = parse_plate(posteriors_with_support)
    assert parsed_supported.text[2] == "0", "O should be converted to 0 when 0 is supported"
    assert len(parsed_supported.corrections) >= 1
    assert any("O to 0" in c.reason for c in parsed_supported.corrections)

    # Case B: Pos 2 is 'O' with ZERO evidence for '0' (must never invent characters!)
    posteriors_without_support = [
        {"M": 0.95}, {"P": 0.95},
        {"O": 0.85, "Q": 0.10},  # Zero probability for '0'
        {"4": 0.95}, {"Z": 0.95}, {"K": 0.95},
        {"6": 0.95}, {"1": 0.95}, {"8": 0.95}, {"4": 0.95}
    ]
    parsed_unsupported = parse_plate(posteriors_without_support)
    assert parsed_unsupported.text[2] == "?", "O must become '?' when 0 has zero evidence"


def test_series_I_and_O_penalized():
    """O or I in the series slot is heavily penalized per CMVR Rule 51(8)."""
    # Compare posterior with 'I' vs standard letter 'E' in series slot
    p_with_I = [
        {"M": 0.95}, {"H": 0.95}, {"1": 0.95}, {"2": 0.95},
        {"I": 0.95}, {"B": 0.95},
        {"1": 0.95}, {"2": 0.95}, {"3": 0.95}, {"4": 0.95}
    ]
    p_with_E = [
        {"M": 0.95}, {"H": 0.95}, {"1": 0.95}, {"2": 0.95},
        {"E": 0.95}, {"B": 0.95},
        {"1": 0.95}, {"2": 0.95}, {"3": 0.95}, {"4": 0.95}
    ]
    parsed_I = parse_plate(p_with_I)
    parsed_E = parse_plate(p_with_E)
    assert any("series_I_penalized" in f for f in parsed_I.flags)
    assert parsed_I.format_confidence < parsed_E.format_confidence


def test_legacy_and_unknown_state_flags():
    """Grandfathered legacy state codes (TS, UA, OR, CT, DN) and unknown codes are flagged."""
    # Legacy state TS
    parsed_legacy = parse_plate_text("TS09AB1234")
    assert any("legacy_state_code_TS" in f for f in parsed_legacy.flags)

    # Unknown state ZZ
    parsed_unknown = parse_plate_text("ZZ09AB1234")
    assert any("unknown_state" in f for f in parsed_unknown.flags)


def test_serial_0000_rejected():
    """Registration serial 0000 is illegal under CMVR Rule 50."""
    parsed = parse_plate_text("WB02XX0000")
    assert parsed.is_valid is False
    assert any("invalid_serial_0000_rejected" in f for f in parsed.flags)


def test_letters_in_number_slot_become_question_mark_unless_supported():
    """Letters in numeric serial slot become '?' unless twin is supported."""
    # Letter 'S' in number slot with no '5' evidence
    posteriors = [
        {"M": 0.95}, {"H": 0.95}, {"1": 0.95}, {"2": 0.95},
        {"A": 0.95}, {"B": 0.95},
        {"1": 0.95}, {"2": 0.95}, {"3": 0.95},
        {"S": 0.80}  # No '5' candidate
    ]
    parsed = parse_plate(posteriors)
    assert parsed.text[-1] == "?", "Unsupported letter in number slot must become '?'"


def test_zero_evidence_produces_all_question_marks():
    """Zero evidence produces all '?' and never invents characters."""
    posteriors = [{} for _ in range(10)]
    parsed = parse_plate(posteriors)
    assert parsed.text == "??????????", "Zero evidence must result in all '?'"
    assert parsed.is_valid is False


def test_special_military_and_diplomatic_formats():
    """Army broad-arrow and diplomatic plates return unrecognised civilian format."""
    parsed_army = parse_plate_text("↑22B137541M")
    assert parsed_army.template == "UNRECOGNISED_FORMAT"
    assert "special_military_format" in parsed_army.flags

    parsed_diplo = parse_plate_text("11CD01")
    assert parsed_diplo.template == "UNRECOGNISED_FORMAT"
    assert "special_diplomatic_format" in parsed_diplo.flags


def test_property_no_unsupported_characters_ever_invented():
    """Property test: the output never contains a character absent from input posteriors."""
    import random
    rng = random.Random(1337)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"

    for _ in range(50):
        length = rng.randint(8, 11)
        posteriors = []
        allowed_chars_per_pos = []
        for _ in range(length):
            k = rng.randint(1, 3)
            chars = rng.sample(alphabet, k)
            probs = [rng.uniform(0.1, 0.9) for _ in range(k)]
            total = sum(probs)
            dist = {c: p / total for c, p in zip(chars, probs)}
            posteriors.append(dist)
            allowed_chars_per_pos.append(set(chars) | {"?"})

        parsed = parse_plate(posteriors)
        for i, ch in enumerate(parsed.text):
            assert ch in allowed_chars_per_pos[i], f"Pos {i}: character '{ch}' was invented without evidence!"


@pytest.mark.parametrize("invalid_plate", [
    "29074W0201",
    "2500451001",
    "H47BL41461",
    "THELICENSE",
    "VEGRESTAURANTIMITIMUIUNIAV",
    "MP04235401",
    "6018AA1001",
    "6A0BAA2LOW"
])
def test_impossible_indian_plate_rejection(invalid_plate: str):
    """Verify that impossible strings, shop signboards, and illegal formats are strictly rejected."""
    from fusion.grammar import is_valid_indian_plate, validate_plate_grammar
    from fusion.vlm_fusion import fuse_ocr_and_vlm

    assert is_valid_indian_plate(invalid_plate) is False
    assert validate_plate_grammar(invalid_plate) is False

    # Fusion with impossible plates must safely abstain to NOT FOUND
    fused, stats = fuse_ocr_and_vlm(None, invalid_plate)
    assert fused.text == "NOT FOUND"
    assert fused.format_valid is False

