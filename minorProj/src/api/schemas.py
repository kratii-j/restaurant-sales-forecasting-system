from pydantic import BaseModel
from typing import Optional, List


# ── Prediction ──────────────────────────────────────────────
class PredictionRequest(BaseModel):
    restaurant_id: int
    date: str
    discount: float = 0.0
    promotion_flag: bool = False


class PredictionResponse(BaseModel):
    restaurant_id: int
    date: str
    predicted_orders: float
    predicted_revenue: float
    lower_bound_orders: float
    upper_bound_orders: float
    lower_bound_revenue: float
    upper_bound_revenue: float
    risk_level: str
    risk_score: float


# ── Dashboard ───────────────────────────────────────────────
class DashboardSummary(BaseModel):
    total_restaurants: int
    avg_daily_orders: float
    avg_daily_revenue: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class TrendPoint(BaseModel):
    date: str
    total_orders: float
    total_revenue: float
    avg_orders: float
    avg_revenue: float


# ── Restaurants ─────────────────────────────────────────────
class RestaurantListItem(BaseModel):
    restaurant_id: int
    restaurant_name: str
    city: str
    cuisines: str
    aggregate_rating: float
    votes: int
    average_cost_for_two: float
    price_range: int
    has_online_delivery: bool
    has_table_booking: bool


class RestaurantDetail(BaseModel):
    restaurant_id: int
    restaurant_name: str
    city: str
    locality: str
    cuisines: str
    aggregate_rating: float
    votes: int
    average_cost_for_two: float
    currency: str
    price_range: int
    has_online_delivery: bool
    has_table_booking: bool
    is_delivering_now: bool
    longitude: float
    latitude: float
    recent_orders: List[dict]


# ── Forecast ────────────────────────────────────────────────
class ForecastPoint(BaseModel):
    date: str
    actual_orders: Optional[float] = None
    predicted_orders: Optional[float] = None
    actual_revenue: Optional[float] = None
    predicted_revenue: Optional[float] = None


# ── Risk ────────────────────────────────────────────────────
class RiskRestaurant(BaseModel):
    restaurant_id: int
    rmse: float
    mae: float
    mean_actual: float
    mean_predicted: float
    risk_score: float
    risk_level: str


# ── Models ──────────────────────────────────────────────────
class ModelComparisonItem(BaseModel):
    model: str
    source: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None