"""Weighted Needleman-Wunsch string alignment using confusable-character penalties."""

from __future__ import annotations

CONFUSION_PAIRS = {
    ("O", "0"), ("0", "O"),
    ("B", "8"), ("8", "B"),
    ("I", "1"), ("1", "I"),
    ("S", "5"), ("5", "S"),
    ("Z", "2"), ("2", "Z"),
    ("D", "0"), ("0", "D"),
    ("G", "6"), ("6", "G"),
    ("T", "7"), ("7", "T"),
    ("V", "U"), ("U", "V"),
}


def substitution_cost(c1: str, c2: str) -> float:
    """Return substitution cost between two characters."""
    if c1 == c2:
        return 0.0
    if (c1, c2) in CONFUSION_PAIRS:
        return 0.35
    return 1.0


def align_sequences(seq1: str, seq2: str, gap_penalty: float = 0.8) -> tuple[str, str]:
    """Align two license plate character strings using Needleman-Wunsch dynamic programming."""
    n, m = len(seq1), len(seq2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(m + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = substitution_cost(seq1[i - 1], seq2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j - 1] + match_cost,
                dp[i - 1][j] + gap_penalty,
                dp[i][j - 1] + gap_penalty,
            )

    # Backtrace
    i, j = n, m
    al1, al2 = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(dp[i][j] - (dp[i - 1][j - 1] + substitution_cost(seq1[i - 1], seq2[j - 1]))) < 1e-5:
            al1.append(seq1[i - 1])
            al2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and abs(dp[i][j] - (dp[i - 1][j] + gap_penalty)) < 1e-5:
            al1.append(seq1[i - 1])
            al2.append("-")
            i -= 1
        else:
            al1.append("-")
            al2.append(seq2[j - 1])
            j -= 1

    return "".join(reversed(al1)), "".join(reversed(al2))
