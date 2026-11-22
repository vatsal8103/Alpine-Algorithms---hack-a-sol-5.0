"""TRACE-X: explainable vehicle shortlisting from partial number plates.

The package is deliberately split into small, testable stages:

    video -> detection -> OCR -> plate fusion -> attributes -> matching -> report

The two stages that must remain team-authored are :mod:`tracex.fusion`
(multi-frame evidence fusion) and :mod:`tracex.matching` (registry ranking).
"""

__version__ = "0.1.0"
