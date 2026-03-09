from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="DineCast API",
    version="1.0.0",
    description="Restaurant Sales Forecasting API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to DineCast API"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DineCast API"
    }