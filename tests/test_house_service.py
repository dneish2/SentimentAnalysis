from analytics.price_stats import PriceStats
from data.listings_repository import ListingsRepository
from services.house_avg_service import HouseAvgService


def test_house_avg_service_handles_missing_data(tmp_path):
    csv_path = tmp_path / "listings.csv"
    csv_path.write_text("neighborhood,bedrooms,bathrooms,sqft,price\n", encoding="utf-8")
    repo = ListingsRepository(str(csv_path))
    service = HouseAvgService(repo, PriceStats())

    result = service.estimate(
        neighborhood="Unknown",
        bedrooms=3,
        bathrooms=2.0,
        sqft=1500,
    )

    assert "error" in result
    assert result["error"]["code"] == "NoData"
