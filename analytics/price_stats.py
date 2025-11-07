"""Statistical utilities for computing comparable listing estimates."""
from __future__ import annotations

import random
from statistics import fmean
from typing import Iterable, List, Sequence, Tuple

from domain.listings import Listing


class PriceStats:
    """Encapsulates comparable filtering and bootstrap confidence intervals."""

    def __init__(self, tol_baths: float = 0.5, tol_sqft: int = 250) -> None:
        self.tol_baths = tol_baths
        self.tol_sqft = tol_sqft

    def filter_comps(
        self,
        listings: Iterable[Listing],
        *,
        bedrooms: int,
        bathrooms: float,
        sqft: int,
    ) -> List[Listing]:
        """Return listings comparable to the supplied spec."""

        comparables: List[Listing] = []
        for listing in listings:
            if listing.bedrooms != bedrooms:
                continue
            if abs(listing.bathrooms - bathrooms) > self.tol_baths:
                continue
            if abs(listing.sqft - sqft) > self.tol_sqft:
                continue
            comparables.append(listing)
        return comparables

    def mean_ci95(self, values: Sequence[float], bootstrap_samples: int = 1000) -> Tuple[float, float, float]:
        """Return the mean and 95% bootstrap confidence interval."""

        if not values:
            return float("nan"), float("nan"), float("nan")
        if len(values) == 1:
            val = float(values[0])
            return val, val, val

        n = len(values)
        samples: List[float] = []
        for _ in range(bootstrap_samples):
            resample = [values[random.randrange(n)] for _ in range(n)]
            samples.append(fmean(resample))
        samples.sort()
        mean_val = fmean(values)
        lower_idx = max(0, int(0.025 * bootstrap_samples) - 1)
        upper_idx = min(len(samples) - 1, int(0.975 * bootstrap_samples) - 1)
        return float(mean_val), float(samples[lower_idx]), float(samples[upper_idx])
