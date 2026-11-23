"""Multi-frame evidential fusion package."""

from fusion.fuser import fuse_frames
from fusion.grammar import is_valid_indian_plate, validate_plate_grammar

__all__ = ["fuse_frames", "is_valid_indian_plate", "validate_plate_grammar"]
