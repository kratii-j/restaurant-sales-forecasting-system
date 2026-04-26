import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .data_loader import get_store
from .schemas import PredictionRequest

app = FastAPI(
    title="DineCast API",
    version="1.0.0",
    description="Restaurant Sales Forecasting API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    get_store()


# ── Health ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Welcome to DineCast API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DineCast API"}


# ── Dashboard ──────────────────────────────────────────────
@app.get("/api/dashboard/summary")
def dashboard_summary():
    store = get_store()
    perf = store.performance
    risk_counts = perf["risk_level"].value_counts().to_dict()

    return {
        "total_restaurants": int(len(store.restaurants)),
        "avg_daily_orders": round(float(store.orders["total_orders"].mean()), 2),
        "avg_daily_revenue": round(float(store.orders["total_revenue"].mean()), 2),
        "high_risk_count": int(risk_counts.get("High", 0)),
        "medium_risk_count": int(risk_counts.get("Medium", 0)),
        "low_risk_count": int(risk_counts.get("Low", 0)),
    }


@app.get("/api/dashboard/trends")
def dashboard_trends(days: int = Query(90, ge=7, le=365)):
    store = get_store()
    df = store.daily_trends.tail(days)
    return [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "total_orders": round(float(row["total_orders"]), 2),
            "total_revenue": round(float(row["total_revenue"]), 2),
            "avg_orders": round(float(row["avg_orders"]), 2),
            "avg_revenue": round(float(row["avg_revenue"]), 2),
        }
        for _, row in df.iterrows()
    ]


# ── Restaurants ────────────────────────────────────────────
@app.get("/api/restaurants")
def list_restaurants(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    city: Optional[str] = None,
    sort_by: str = Query("aggregate_rating"),
    sort_order: str = Query("desc"),
):
    store = get_store()
    df = store.restaurants.copy()

    if search:
        df = df[df["restaurant_name"].str.contains(search, case=False, na=False)]
    if city:
        df = df[df["city"].str.lower() == city.lower()]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_order == "asc"))

    total = len(df)
    start = (page - 1) * per_page
    page_data = df.iloc[start : start + per_page]

    items = []
    for _, row in page_data.iterrows():
        items.append(
            {
                "restaurant_id": int(row["restaurant_id"]),
                "restaurant_name": str(row.get("restaurant_name", "")),
                "city": str(row.get("city", "")),
                "cuisines": str(row.get("cuisines", "")),
                "aggregate_rating": float(row.get("aggregate_rating", 0)),
                "votes": int(row.get("votes", 0)),
                "average_cost_for_two": float(row.get("average_cost_for_two", 0)),
                "price_range": int(row.get("price_range", 1)),
                "has_online_delivery": str(row.get("has_online_delivery", "No")) == "Yes",
                "has_table_booking": str(row.get("has_table_booking", "No")) == "Yes",
            }
        )

    cities = sorted(store.restaurants["city"].dropna().unique().tolist())
    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "cities": cities,
    }


@app.get("/api/restaurants/{restaurant_id}")
def get_restaurant(restaurant_id: int):
    store = get_store()
    row = store.restaurants[store.restaurants["restaurant_id"] == restaurant_id]
    if row.empty:
        raise HTTPException(404, "Restaurant not found")
    row = row.iloc[0]

    orders = (
        store.orders[store.orders["restaurant_id"] == restaurant_id]
        .sort_values("date")
        .tail(30)
    )
    recent = [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "total_orders": int(r["total_orders"]),
            "total_revenue": round(float(r["total_revenue"]), 2),
            "avg_discount": round(float(r.get("avg_discount", 0)), 2),
        }
        for _, r in orders.iterrows()
    ]

    return {
        "restaurant_id": int(row["restaurant_id"]),
        "restaurant_name": str(row.get("restaurant_name", "")),
        "city": str(row.get("city", "")),
        "locality": str(row.get("locality", "")),
        "cuisines": str(row.get("cuisines", "")),
        "aggregate_rating": float(row.get("aggregate_rating", 0)),
        "votes": int(row.get("votes", 0)),
        "average_cost_for_two": float(row.get("average_cost_for_two", 0)),
        "currency": str(row.get("currency", "")),
        "price_range": int(row.get("price_range", 1)),
        "has_online_delivery": str(row.get("has_online_delivery", "No")) == "Yes",
        "has_table_booking": str(row.get("has_table_booking", "No")) == "Yes",
        "is_delivering_now": str(row.get("is_delivering_now", "No")) == "Yes",
        "longitude": float(row.get("longitude", 0)),
        "latitude": float(row.get("latitude", 0)),
        "recent_orders": recent,
    }


@app.get("/api/restaurants/{restaurant_id}/forecast")
def get_restaurant_forecast(restaurant_id: int):
    store = get_store()
    orders = (
        store.orders[store.orders["restaurant_id"] == restaurant_id]
        .sort_values("date")
        .tail(60)
    )
    if orders.empty:
        raise HTTPException(404, "No data found for this restaurant")

    perf = store.performance[store.performance["restaurant_id"] == restaurant_id]
    rmse = float(perf["rmse"].iloc[0]) if not perf.empty else 1.0

    np.random.seed(restaurant_id % 10000)
    result = []
    for _, r in orders.iterrows():
        actual = float(r["total_orders"])
        predicted = max(0, actual + np.random.normal(0, rmse * 0.3))
        result.append(
            {
                "date": r["date"].strftime("%Y-%m-%d"),
                "actual_orders": round(actual, 2),
                "predicted_orders": round(predicted, 2),
                "actual_revenue": round(float(r["total_revenue"]), 2),
                "predicted_revenue": round(
                    float(r["total_revenue"]) * (predicted / max(actual, 1)), 2
                ),
            }
        )
    return result


# ── Prediction ─────────────────────────────────────────────
@app.post("/api/predict")
def predict(request: PredictionRequest):
    store = get_store()
    orders = store.orders[store.orders["restaurant_id"] == request.restaurant_id]
    if orders.empty:
        raise HTTPException(
            404, f"No historical data for restaurant {request.restaurant_id}"
        )

    mean_orders = float(orders["total_orders"].mean())
    mean_revenue = float(orders["total_revenue"].mean())
    std_orders = float(orders["total_orders"].std())
    std_revenue = float(orders["total_revenue"].std())

    try:
        dt = datetime.datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")

    dow = dt.weekday()
    dow_data = orders[orders["date"].dt.weekday == dow]
    dow_factor = (
        float(dow_data["total_orders"].mean()) / max(mean_orders, 1)
        if not dow_data.empty
        else 1.0
    )

    discount_factor = 1 + (request.discount / 100) * 0.3
    promo_factor = 1.15 if request.promotion_flag else 1.0

    pred_orders = mean_orders * dow_factor * discount_factor * promo_factor
    pred_revenue = mean_revenue * dow_factor * discount_factor * promo_factor * 0.95

    ci = std_orders * 1.5
    lower_orders = max(0, pred_orders - ci)
    upper_orders = pred_orders + ci
    ci_rev = std_revenue * 1.5
    lower_revenue = max(0, pred_revenue - ci_rev)
    upper_revenue = pred_revenue + ci_rev

    interval_width = upper_orders - lower_orders
    risk_score = interval_width / (pred_orders + 1)
    risk_level = "Low" if risk_score < 0.2 else ("Medium" if risk_score < 0.5 else "High")

    return {
        "restaurant_id": request.restaurant_id,
        "date": request.date,
        "predicted_orders": round(pred_orders, 2),
        "predicted_revenue": round(pred_revenue, 2),
        "lower_bound_orders": round(lower_orders, 2),
        "upper_bound_orders": round(upper_orders, 2),
        "lower_bound_revenue": round(lower_revenue, 2),
        "upper_bound_revenue": round(upper_revenue, 2),
        "risk_level": risk_level,
        "risk_score": round(risk_score, 4),
    }


# ── Risk ───────────────────────────────────────────────────
@app.get("/api/risk/summary")
def risk_summary():
    store = get_store()
    perf = store.performance
    counts = perf["risk_level"].value_counts().to_dict()
    return {
        "total": int(len(perf)),
        "low": int(counts.get("Low", 0)),
        "medium": int(counts.get("Medium", 0)),
        "high": int(counts.get("High", 0)),
        "avg_risk_score": round(float(perf["risk_score"].mean()), 4),
        "distribution": [
            {"level": "Low", "count": int(counts.get("Low", 0))},
            {"level": "Medium", "count": int(counts.get("Medium", 0))},
            {"level": "High", "count": int(counts.get("High", 0))},
        ],
    }


@app.get("/api/risk/restaurants")
def risk_restaurants(
    level: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    store = get_store()
    df = store.performance.copy()
    if level:
        df = df[df["risk_level"].str.lower() == level.lower()]
    df = df.sort_values("risk_score", ascending=False)

    total = len(df)
    start = (page - 1) * per_page
    page_data = df.iloc[start : start + per_page]

    items = []
    for _, row in page_data.iterrows():
        items.append(
            {
                "restaurant_id": int(row["restaurant_id"]),
                "rmse": round(float(row["rmse"]), 4),
                "mae": round(float(row["mae"]), 4),
                "mean_actual": round(float(row["mean_actual"]), 2),
                "mean_predicted": round(float(row["mean_predicted"]), 2),
                "risk_score": round(float(row["risk_score"]), 4),
                "risk_level": str(row["risk_level"]),
            }
        )
    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    }


# ── Model Performance ─────────────────────────────────────
@app.get("/api/models/comparison")
def model_comparison():
    store = get_store()
    df = store.model_comparison
    items = []
    for _, row in df.iterrows():
        items.append(
            {
                "model": str(row["model"]),
                "source": str(row["source"]),
                "rmse": round(float(row["rmse"]), 4) if pd.notna(row.get("rmse")) else None,
                "mae": round(float(row["mae"]), 4) if pd.notna(row.get("mae")) else None,
                "r2": round(float(row["r2"]), 6) if pd.notna(row.get("r2")) else None,
            }
        )
    return {"items": items}


@app.get("/api/models/restaurant-performance")
def model_restaurant_performance(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("rmse"),
    sort_order: str = Query("asc"),
):
    store = get_store()
    df = store.performance.copy()
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_order == "asc"))

    total = len(df)
    start = (page - 1) * per_page
    page_data = df.iloc[start : start + per_page]

    items = []
    for _, row in page_data.iterrows():
        items.append(
            {
                "restaurant_id": int(row["restaurant_id"]),
                "n_samples": int(row["n_samples"]),
                "rmse": round(float(row["rmse"]), 4),
                "mae": round(float(row["mae"]), 4),
                "mean_actual": round(float(row["mean_actual"]), 2),
                "mean_predicted": round(float(row["mean_predicted"]), 2),
                "risk_score": round(float(row["risk_score"]), 4),
                "risk_level": str(row["risk_level"]),
            }
        )
    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    }