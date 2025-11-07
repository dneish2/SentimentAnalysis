"""Repository for accessing housing listings stored in CSV files."""
from __future__ import annotations

import csv
from functools import lru_cache
from typing import Iterable, List

from domain.listings import Listing


class ListingsRepository:
    """Load listings from a CSV file into domain objects."""

    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path

    @lru_cache(maxsize=1)
    def all(self) -> List[Listing]:
        """Return all listings from the CSV file.

        Invalid rows are skipped to maximise robustness when handling
        heterogeneous or partially missing data.
        """

        listings: List[Listing] = []
        with open(self._csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    listings.append(
                        Listing(
                            neighborhood=str(row["neighborhood"]).strip(),
                            bedrooms=int(row["bedrooms"]),
                            bathrooms=float(row["bathrooms"]),
                            sqft=int(row["sqft"]),
                            price=float(row["price"]),
                        )
                    )
                except (TypeError, ValueError, KeyError):
                    # Skip malformed rows without failing the entire request.
                    continue
        return listings

    def by_neighborhood(self, name: str) -> Iterable[Listing]:
        """Yield listings that match the requested neighbourhood."""

        normalized = name.strip().lower()
        return [listing for listing in self.all() if listing.neighborhood.strip().lower() == normalized]
