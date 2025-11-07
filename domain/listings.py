"""Domain models for housing listings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Listing:
    """Immutable representation of a housing listing."""

    neighborhood: str
    bedrooms: int
    bathrooms: float
    sqft: int
    price: float
    year_built: Optional[int] = None
    renovated: Optional[bool] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
