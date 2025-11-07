"""Application service for housing estimates."""
from __future__ import annotations

from typing import Any, Dict

from analytics.price_stats import PriceStats
from data.listings_repository import ListingsRepository


class HouseAvgService:
    """Coordinate data access and analytics for housing estimates."""

    def __init__(self, repository: ListingsRepository, stats: PriceStats) -> None:
        self._repository = repository
        self._stats = stats

    def estimate(
        self,
        *,
        neighborhood: str,
        bedrooms: int,
        bathrooms: float,
        sqft: int,
        bootstrap_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Return an estimate payload for the requested specification."""

        all_listings = self._repository.by_neighborhood(neighborhood)
        comparables = self._stats.filter_comps(
            all_listings, bedrooms=bedrooms, bathrooms=bathrooms, sqft=sqft
        )
        prices = [listing.price for listing in comparables]
        if not prices:
            return {
                "error": {
                    "code": "NoData",
                    "message": "No comparable listings matched the supplied specification.",
                },
                "cohort": {
                    "count": 0,
                    "neighborhood": neighborhood,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "sqft": sqft,
                },
            }

        mean_val, lower, upper = self._stats.mean_ci95(prices, bootstrap_samples=bootstrap_samples)
        return {
            "estimation": {
                "mean": round(mean_val, 0),
                "ci95": [round(lower, 0), round(upper, 0)],
                "unit": "usd",
                "explanation": (
                    "Bootstrap 95% CI using listings in the same neighborhood with comparable"
                    " bedrooms, bathrooms, and square footage."
                ),
            },
            "cohort": {
                "count": len(prices),
                "neighborhood": neighborhood,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft": sqft,
                "price_examples": prices[:5],
            },
        }
