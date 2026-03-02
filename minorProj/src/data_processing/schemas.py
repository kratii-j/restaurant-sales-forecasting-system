"""
schemas.py
~~~~~~~~~~
Pydantic models for the three core datasets used in minorProj.

    1. RestaurantMaster  – static restaurant-level attributes
    2. DailyOrders       – daily time-series target data
    3. ExternalContext    – external / calendar context features

Each model enforces type constraints, value ranges, and business-logic
validations so that noisy or malformed records are caught early in the
data pipeline.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ── Enums ────────────────────────────────────────────────────

class PriceRange(int, Enum):
    """Restaurant pricing tier (1 = budget … 4 = premium)."""
    BUDGET = 1
    MID = 2
    HIGH = 3
    PREMIUM = 4


class RiskLevel(str, Enum):
    """Demand risk classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class WeatherCondition(str, Enum):
    """Simplified weather categories."""
    CLEAR = "Clear"
    CLOUDY = "Cloudy"
    RAINY = "Rainy"
    STORMY = "Stormy"
    SNOWY = "Snowy"
    UNKNOWN = "Unknown"


# ══════════════════════════════════════════════════════════════
# Dataset 1 — Restaurant Master (Static Features)
# ══════════════════════════════════════════════════════════════

class RestaurantMaster(BaseModel):
    """
    Static restaurant-level attributes.

    Captures restaurant positioning, cuisine-specific demand behavior,
    pricing tier, quality indicators, and geographic location.
    """

    restaurant_id: int = Field(
        ..., gt=0,
        description="Unique restaurant identifier (Primary Key).",
    )
    restaurant_name: str = Field(
        ..., min_length=1, max_length=256,
        description="Display name of the restaurant.",
    )
    country_code: int = Field(
        ..., ge=1,
        description="Numeric ISO country code.",
    )
    city: str = Field(
        ..., min_length=1, max_length=128,
        description="City where the restaurant is located.",
    )
    locality: str = Field(
        default="", max_length=256,
        description="Locality / neighbourhood within the city.",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Geographic longitude.",
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Geographic latitude.",
    )
    cuisines: str = Field(
        ..., min_length=1,
        description="Comma-separated list of cuisines offered.",
    )
    average_cost_for_two: float = Field(
        ..., ge=0.0,
        description="Average cost for two people in local currency.",
    )
    currency: str = Field(
        ..., min_length=1, max_length=16,
        description="Currency code (e.g. INR, USD).",
    )
    has_table_booking: bool = Field(
        default=False,
        description="Whether the restaurant supports table booking.",
    )
    has_online_delivery: bool = Field(
        default=False,
        description="Whether the restaurant supports online delivery.",
    )
    is_delivering_now: bool = Field(
        default=False,
        description="Whether the restaurant is currently delivering.",
    )
    switch_to_order_menu: bool = Field(
        default=False,
        description="Whether the user can switch to the order menu.",
    )
    price_range: PriceRange = Field(
        ...,
        description="Pricing tier (1=Budget, 2=Mid, 3=High, 4=Premium).",
    )
    aggregate_rating: float = Field(
        ..., ge=0.0, le=5.0,
        description="Overall restaurant rating (0.0 – 5.0).",
    )
    votes: int = Field(
        ..., ge=0,
        description="Total number of customer votes / reviews.",
    )

    # ── Validators ───────────────────────────────────────────

    @field_validator("cuisines")
    @classmethod
    def validate_cuisines(cls, v: str) -> str:
        """Strip whitespace from each cuisine and remove empties."""
        cleaned = [c.strip() for c in v.split(",") if c.strip()]
        if not cleaned:
            raise ValueError("At least one cuisine must be provided.")
        return ", ".join(cleaned)

    @field_validator("city", "locality")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    class Config:
        use_enum_values = True
        str_strip_whitespace = True


# ══════════════════════════════════════════════════════════════
# Dataset 2 — Daily Orders (Time-Series Target Data)
# ══════════════════════════════════════════════════════════════

class DailyOrders(BaseModel):
    """
    One record per restaurant per day.

    Provides the two forecasting targets (total_orders, total_revenue)
    together with auxiliary operational metrics used as features.
    """

    restaurant_id: int = Field(
        ..., gt=0,
        description="Foreign key → RestaurantMaster.restaurant_id.",
    )
    date: date = Field(
        ...,
        description="Calendar date for the observation (YYYY-MM-DD).",
    )
    total_orders: int = Field(
        ..., ge=0,
        description="Total orders placed on this date (Target 1).",
    )
    total_revenue: float = Field(
        ..., ge=0.0,
        description="Total revenue earned on this date (Target 2).",
    )
    avg_discount: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Average discount percentage applied (0–100).",
    )
    promotion_flag: bool = Field(
        default=False,
        description="Whether a promotion was active on this date.",
    )
    cancellation_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Order cancellation rate (0.0–1.0), optional.",
    )
    avg_delivery_time: Optional[float] = Field(
        default=None, ge=0.0,
        description="Average delivery time in minutes, optional.",
    )

    # ── Validators ───────────────────────────────────────────

    @field_validator("date")
    @classmethod
    def date_not_in_future(cls, v: date) -> date:
        """Historical data should not contain future dates."""
        if v > date.today():
            raise ValueError(
                f"Date {v} is in the future. "
                "Only historical records are accepted."
            )
        return v

    @model_validator(mode="after")
    def revenue_consistent_with_orders(self):
        """Revenue should be zero when there are no orders."""
        if self.total_orders == 0 and self.total_revenue > 0:
            raise ValueError(
                "total_revenue must be 0 when total_orders is 0."
            )
        return self

    class Config:
        use_enum_values = True


# ══════════════════════════════════════════════════════════════
# Dataset 3 — External Context
# ══════════════════════════════════════════════════════════════

class ExternalContext(BaseModel):
    """
    Calendar and external contextual features for a given date + city.

    Used to capture seasonality and external demand drivers such as
    holidays, weather, and local events.
    """

    date: date = Field(
        ...,
        description="Calendar date (YYYY-MM-DD).",
    )
    city: str = Field(
        ..., min_length=1, max_length=128,
        description="City to which this context applies.",
    )
    day_of_week: int = Field(
        ..., ge=0, le=6,
        description="Day of week (0=Monday … 6=Sunday).",
    )
    is_weekend: bool = Field(
        ...,
        description="True if Saturday or Sunday.",
    )
    is_holiday: bool = Field(
        default=False,
        description="True if the date is a public holiday.",
    )
    month: int = Field(
        ..., ge=1, le=12,
        description="Month of the year (1–12).",
    )
    weather: WeatherCondition = Field(
        default=WeatherCondition.UNKNOWN,
        description="Simplified weather category for the day.",
    )
    event_flag: bool = Field(
        default=False,
        description="True if a notable local event is occurring.",
    )

    # ── Validators ───────────────────────────────────────────

    @field_validator("city")
    @classmethod
    def strip_city(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def validate_day_of_week(self):
        """Ensure day_of_week matches the actual date."""
        expected = self.date.weekday()  # Monday=0 … Sunday=6
        if self.day_of_week != expected:
            raise ValueError(
                f"day_of_week={self.day_of_week} does not match "
                f"date={self.date} (expected {expected})."
            )
        return self

    @model_validator(mode="after")
    def validate_is_weekend(self):
        """Ensure is_weekend is consistent with day_of_week."""
        expected = self.day_of_week >= 5  # Sat=5, Sun=6
        if self.is_weekend != expected:
            raise ValueError(
                f"is_weekend={self.is_weekend} is inconsistent with "
                f"day_of_week={self.day_of_week}."
            )
        return self

    @model_validator(mode="after")
    def validate_month(self):
        """Ensure month matches the actual date."""
        if self.month != self.date.month:
            raise ValueError(
                f"month={self.month} does not match "
                f"date={self.date} (expected {self.date.month})."
            )
        return self

    class Config:
        use_enum_values = True
        str_strip_whitespace = True
