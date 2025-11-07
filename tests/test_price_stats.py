from analytics.price_stats import PriceStats
from domain.listings import Listing


def test_filter_comps_matches_tolerances():
    listings = [
        Listing("A", 3, 2.0, 1500, 300000),
        Listing("A", 3, 2.7, 1500, 400000),
        Listing("A", 2, 2.0, 1500, 250000),
        Listing("A", 3, 2.2, 1700, 320000),
    ]
    stats = PriceStats(tol_baths=0.5, tol_sqft=250)
    comps = stats.filter_comps(listings, bedrooms=3, bathrooms=2.0, sqft=1500)
    assert len(comps) == 2
    assert {c.price for c in comps} == {300000, 320000}


def test_mean_ci95_handles_single_value():
    stats = PriceStats()
    mean_val, lower, upper = stats.mean_ci95([350000], bootstrap_samples=100)
    assert mean_val == lower == upper == 350000.0
