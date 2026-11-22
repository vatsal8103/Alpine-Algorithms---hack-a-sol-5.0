"""Indian number-plate grammar, alignment and normalisation.

The important rule in this module is: **normalisation may reinterpret an
observed character, but it may never create one.**  Every non-``?`` output
character has to come from an OCR reading (or an explicit investigator hint).
Confusion-pair rewrites such as ``0 -> O`` in a letter slot are recorded in a
``corrections`` list and slightly reduce confidence.

Supported grammar
-----------------
Standard Indian plates::

    SS RR LLL NNNN      e.g. JH 05 AB 6799
    SS R  LLL NNNN      e.g. MH 1 AB 1234   (older / single-digit RTO)

where ``SS`` = state, ``RR`` = RTO, ``LLL`` = 1-3 series letters and ``NNNN``
= 4 digits.  BH-series plates (``22 BH 1234 AA``) are also recognised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ALNUM = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")
UNKNOWN = "?"

# Characters that OCR routinely confuses.  This table is used for *scoring*
# and for class-aware normalisation; it is never used to fill an unknown slot.
CONFUSION: dict[str, set[str]] = {
    "0": set("OQD"),
    "O": set("0QD"),
    "1": set("ILT"),
    "I": set("1LT"),
    "2": set("Z"),
    "Z": set("2"),
    "5": set("S"),
    "S": set("5"),
    "8": set("B"),
    "B": set("8"),
    "6": set("G"),
    "G": set("6"),
    "4": set("A"),
    "A": set("4"),
    "7": set("T"),
    "T": set("7"),
    "9": set("G"),
    "U": set("V"),
    "V": set("U"),
    "M": set("N"),
    "N": set("M"),
}

# Rule-based corrections used only when the plate grammar says the slot must
# be a digit/letter.  Each entry maps the observed character to the expected
# class equivalent.
_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "T": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "B": "8",
}
_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
}


def clean_text(text: str | None) -> str:
    """Uppercase and keep only plate-legal characters plus ``?``.

    Common separators and OCR noise (``-``, spaces, ``|``) are removed.  The
    literal ``IND`` strip is applied only if the remaining string still looks
    like a plate, so a plate ending in ``IND`` is not accidentally damaged.
    """

    if not text:
        return ""
    value = str(text).upper().strip()
    value = value.replace("|", "I").replace("—", "").replace("–", "")
    value = re.sub(r"[^A-Z0-9?]", "", value)
    if "IND" in value and len(value) > 10:
        stripped = value.replace("IND", "")
        if len(stripped) >= 7:
            value = stripped
    return value


def normalize_hint(text: str | None) -> str:
    """Normalise an investigator-provided partial plate.

    ``*`` and ``_`` are treated as explicitly unknown positions.
    """

    if not text:
        return ""
    return clean_text(text.replace("*", UNKNOWN).replace("_", UNKNOWN))


def is_letter(ch: str) -> bool:
    return ch in LETTERS


def is_digit(ch: str) -> bool:
    return ch in DIGITS


def is_bh_series(text: str) -> bool:
    return len(text) >= 4 and text[2:4] == "BH"


def class_pattern(length: int, rto_digits: int = 2, bh: bool = False) -> str:
    """Return a class template such as ``LLDDLLDDDD``.

    ``L`` = letter, ``D`` = digit.  A ``?`` class is used for unsupported
    lengths rather than pretending to know the grammar.
    """

    if bh:
        # 22 BH 1234 AA  -> D D L L D D D D L L
        series = max(0, length - 10)
        if 10 <= length <= 12:
            return "DDLL" + "DDDD" + "L" * series
        return "?" * length

    if length < 7 or length > 11:
        return "?" * length
    if rto_digits not in (1, 2):
        rto_digits = 2
    series = length - 2 - rto_digits - 4
    if series < 0 or series > 3:
        return "?" * length
    return "LL" + "D" * rto_digits + "L" * series + "DDDD"


def candidate_patterns(length: int) -> list[tuple[str, int, bool]]:
    """All plausible grammar hypotheses for a given length.

    Returns ``(pattern, rto_digits, bh)`` tuples.  BH is only considered when
    the reading itself contains ``BH``; otherwise it would be too permissive.
    """

    patterns: list[tuple[str, int, bool]] = []
    for rto in (2, 1):
        p = class_pattern(length, rto_digits=rto)
        if "?" not in p:
            patterns.append((p, rto, False))
    return patterns or [(class_pattern(length), 2, False)]


def infer_pattern(text: str) -> str:
    """Best-effort grammar pattern for a complete/partial plate string."""

    cleaned = clean_text(text)
    if is_bh_series(cleaned):
        return class_pattern(len(cleaned), bh=True)
    return class_pattern(len(cleaned), rto_digits=2)


def char_matches_class(ch: str, expected: str) -> bool:
    if ch == UNKNOWN or expected == "?":
        return True
    if expected == "L":
        return ch in LETTERS
    if expected == "D":
        return ch in DIGITS
    return True


def confusion_match(a: str, b: str) -> bool:
    return a != UNKNOWN and b != UNKNOWN and b in CONFUSION.get(a, set())


def class_match_score(ch: str, expected: str) -> float:
    """How compatible is an observed character with a grammar slot?

    1.0 exact class, ~0.65 for a known OCR confusion, 0.0 for a hard class
    violation (e.g. ``A`` in a digit slot).
    """

    if ch == UNKNOWN:
        return 0.35
    if expected == "?":
        return 0.5
    if char_matches_class(ch, expected):
        return 1.0
    if expected == "D" and ch in _TO_DIGIT:
        return 0.65
    if expected == "L" and ch in _TO_LETTER:
        return 0.65
    return 0.0


def correct_for_class(ch: str, expected: str) -> tuple[str, str | None]:
    """Reinterpret one observed character using plate grammar.

    Returns ``(character, rule)``.  ``rule`` is ``None`` when no rewrite
    happened.  The caller records the rule and applies a small confidence
    penalty; the character itself is always derived from an observation.
    """

    if ch == UNKNOWN or expected == "?" or char_matches_class(ch, expected):
        return ch, None
    if expected == "D" and ch in _TO_DIGIT:
        return _TO_DIGIT[ch], f"{ch}->{_TO_DIGIT[ch]} in digit slot"
    if expected == "L" and ch in _TO_LETTER:
        return _TO_LETTER[ch], f"{ch}->{_TO_LETTER[ch]} in letter slot"
    return ch, None


@dataclass
class AlignedReading:
    """A raw reading mapped onto a fixed-length plate template."""

    text: str
    chars: list[str]
    confidences: list[float]
    raw_chars: list[str]
    corrections: list[dict] = field(default_factory=list)
    match_score: float = 0.0
    pattern: str = ""

    @property
    def known_chars(self) -> int:
        return sum(1 for c in self.chars if c != UNKNOWN)


def _as_conf_list(confidences: list[float] | float | None, n: int, default: float) -> list[float]:
    if isinstance(confidences, (int, float)):
        return [float(confidences)] * n
    if not confidences:
        return [default] * n
    values = [float(x) for x in confidences]
    if len(values) < n:
        values.extend([default] * (n - len(values)))
    return values[:n]


def align_reading(
    reading: str,
    length: int,
    pattern: str | None = None,
    char_confidences: list[float] | float | None = None,
    line_confidence: float = 0.5,
    class_correction: bool = True,
) -> AlignedReading:
    """Align a possibly truncated reading to a fixed plate length.

    Uses Needleman-Wunsch with asymmetric costs:

    * a template slot with no observed character costs ``0.35``;
    * an extra observed character (noise outside the plate) costs ``0.90``;
    * a substitution costs ``1 - class_match_score``.

    The gap cost is deliberately lower than a hard class violation, so a
    suffix-only reading such as ``AB6799`` is placed at the *end* of the
    template instead of being stretched across the state/RTO slots.
    """

    raw = clean_text(reading)
    raw = raw.replace("*", UNKNOWN).replace("_", UNKNOWN)
    n = len(raw)
    if pattern is None:
        pattern = class_pattern(length)
    L = min(len(pattern), length)
    pattern = pattern[:L]
    confs = _as_conf_list(char_confidences, n, line_confidence)

    if n == 0:
        return AlignedReading(
            text=UNKNOWN * L,
            chars=[UNKNOWN] * L,
            confidences=[0.0] * L,
            raw_chars=[UNKNOWN] * L,
            match_score=0.0,
            pattern=pattern,
        )

    gap_cost = 0.35
    insert_cost = 0.90
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(L + 1)]
    choice = [[""] * (n + 1) for _ in range(L + 1)]
    dp[0][0] = 0.0
    for i in range(1, L + 1):
        dp[i][0] = i * gap_cost
        choice[i][0] = "gap"
    for j in range(1, n + 1):
        dp[0][j] = j * insert_cost
        choice[0][j] = "insert"

    for i in range(1, L + 1):
        expected = pattern[i - 1]
        for j in range(1, n + 1):
            ch = raw[j - 1]
            sub_cost = dp[i - 1][j - 1] + (1.0 - class_match_score(ch, expected))
            gap_cost_here = dp[i - 1][j] + gap_cost
            ins_cost_here = dp[i][j - 1] + insert_cost
            best = min(sub_cost, gap_cost_here, ins_cost_here)
            dp[i][j] = best
            # Prefer a real match, then a template gap, then a noisy insertion.
            if abs(best - sub_cost) < 1e-9:
                choice[i][j] = "sub"
            elif abs(best - gap_cost_here) < 1e-9:
                choice[i][j] = "gap"
            else:
                choice[i][j] = "insert"

    chars = [UNKNOWN] * L
    aligned_conf = [0.0] * L
    raw_chars = [UNKNOWN] * L
    corrections: list[dict] = []
    score_num = 0.0
    score_den = 0.0
    i, j = L, n
    while i > 0 or j > 0:
        action = choice[i][j]
        if action == "sub":
            observed = raw[j - 1]
            expected = pattern[i - 1]
            conf = max(0.0, min(1.0, confs[j - 1]))
            final, rule = (observed, None)
            if class_correction:
                final, rule = correct_for_class(observed, expected)
            chars[i - 1] = final
            raw_chars[i - 1] = observed
            aligned_conf[i - 1] = conf * (0.92 if rule else 1.0)
            if rule:
                corrections.append(
                    {
                        "position": i - 1,
                        "observed": observed,
                        "corrected": final,
                        "rule": rule,
                        "confidence": round(aligned_conf[i - 1], 4),
                    }
                )
            if observed != UNKNOWN:
                score_num += class_match_score(observed, expected) * max(conf, 0.05)
                score_den += max(conf, 0.05)
            i -= 1
            j -= 1
        elif action == "gap":
            i -= 1
        else:  # insert: observed character outside the template
            j -= 1

    match_score = score_num / score_den if score_den else 0.0
    return AlignedReading(
        text="".join(chars),
        chars=chars,
        confidences=aligned_conf,
        raw_chars=raw_chars,
        corrections=corrections,
        match_score=match_score,
        pattern=pattern,
    )


def plate_likeness(text: str) -> float:
    """Heuristic 0-1 score used to choose between competing OCR lines."""

    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    known = [c for c in cleaned if c != UNKNOWN]
    if len(known) < 4:
        return 0.0
    pattern = infer_pattern(cleaned)
    if "?" not in pattern:
        compatible = sum(1 for c, p in zip(cleaned, pattern) if char_matches_class(c, p))
        grammar = compatible / len(pattern)
    else:
        grammar = 0.5
    length_score = 1.0 if 9 <= len(cleaned) <= 11 else (0.75 if len(cleaned) == 8 else 0.4)
    coverage = min(1.0, len(known) / 8.0)
    return max(0.0, min(1.0, 0.40 * grammar + 0.30 * length_score + 0.30 * coverage))


def is_valid_plate(text: str) -> bool:
    """True if all known characters fit the Indian plate grammar.

    Both 2-digit and 1-digit RTO layouts are accepted, plus BH-series.
    """

    cleaned = clean_text(text)
    if len(cleaned) < 7 or len(cleaned) > 11:
        return False
    if is_bh_series(cleaned):
        patterns = [class_pattern(len(cleaned), bh=True)]
    else:
        patterns = [
            class_pattern(len(cleaned), rto_digits=2),
            class_pattern(len(cleaned), rto_digits=1),
        ]
    for pattern in patterns:
        if "?" in pattern:
            continue
        if all(char_matches_class(c, p) for c, p in zip(cleaned, pattern)):
            return True
    return False


def format_plate_display(text: str) -> str:
    """Group a plate for human display, preserving ``?`` positions."""

    cleaned = clean_text(text)
    if is_bh_series(cleaned):
        return f"{cleaned[:2]} BH {cleaned[4:8]} {cleaned[8:]}".strip()
    # Single-digit RTO: position 2 is a digit and position 3 is a letter.
    if len(cleaned) >= 8 and cleaned[2].isdigit() and cleaned[3].isalpha():
        series_len = max(0, len(cleaned) - 7)
        return f"{cleaned[:2]} {cleaned[2:3]} {cleaned[3:3 + series_len]} {cleaned[3 + series_len:]}".strip()
    if len(cleaned) >= 9:
        # SS RR [series] NNNN
        series_len = len(cleaned) - 8
        return f"{cleaned[:2]} {cleaned[2:4]} {cleaned[4:4 + series_len]} {cleaned[4 + series_len:]}".strip()
    if len(cleaned) == 8:
        return f"{cleaned[:2]} {cleaned[2:4]} {cleaned[4:]}"
    if len(cleaned) == 7:
        return f"{cleaned[:2]} {cleaned[2:3]} {cleaned[3:]}"
    return cleaned


def pattern_string(text: str) -> str:
    return "".join("L" if c in LETTERS else "D" if c in DIGITS else UNKNOWN for c in clean_text(text))


def mask_for_hint(hint: str) -> str:
    """Return a regex-like mask where known hint chars are fixed."""

    cleaned = normalize_hint(hint)
    return "".join(c if c != UNKNOWN else "." for c in cleaned)
