"""Integrity validation for dense embedding vectors."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

from .providers.base import Vector


class VectorValidationError(ValueError):
    """Raised when an embedding vector fails integrity validation."""


class VectorValidator:
    """Validate vector dimensions and numeric integrity."""

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Expected vector dimension."""
        return self._dimensions

    def validate(self, vector: Sequence[float]) -> Vector:
        """Return an immutable vector after validating its shape and values."""
        if len(vector) != self._dimensions:
            raise VectorValidationError(
                f"Expected {self._dimensions} dimensions, received {len(vector)}"
            )

        normalized = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise VectorValidationError("Vector values must be numeric")
            float_value = float(value)
            if not isfinite(float_value):
                raise VectorValidationError("Vector values must be finite")
            normalized.append(float_value)
        return tuple(normalized)

