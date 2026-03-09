from pydantic import BaseModel


class PredictionRequest(BaseModel):
    restaurant_id: str
    date: str
    discount: float
    promotion_flag: bool


class PredictionResponse(BaseModel):
    restaurant_id: str
    date: str
    predicted_orders: float
    predicted_revenue: float
    lower_bound_orders: float
    upper_bound_orders: float
    lower_bound_revenue: float
    upper_bound_revenue: float
    risk_level: str
    risk_score: float