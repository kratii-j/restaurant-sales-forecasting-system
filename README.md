# 🍽️ DineCast — Restaurant Sales Forecasting System

A full-stack analytics platform for restaurant chains to forecast daily sales, assess operational risk, and visualise model performance. Built with **FastAPI** (Python) on the backend and **React + TypeScript** (Vite) on the frontend.

---

## Project Structure


restaurant-sales-forecasting-system/
├── .venv/                        # Root-level virtual environment (optional)
└── minorProj/
    ├── .env                      # Environment variables (never commit this)
    ├── requirements.txt          # Python dependencies
    ├── config/                   # App & model configuration files
    ├── data/                     # Raw & processed datasets
    ├── docs/                     # Project documentation
    ├── notebooks/                # Jupyter notebooks for EDA & model training
    ├── tests/                    # Backend unit & integration tests
    ├── src/
    │   ├── __init__.py
    │   ├── api/
    │   │   ├── main.py           # FastAPI application & all route definitions
    │   │   ├── data_loader.py    # Data ingestion & in-memory store
    │   │   └── schemas.py        # Pydantic request/response models
    │   ├── data_processing/      # ETL & feature engineering pipelines
    │   ├── features/             # Feature builders
    │   ├── models/               # Trained model artefacts & training scripts
    │   └── utils/                # Shared helper utilities
    └── frontend/
        ├── index.html
        ├── package.json
        ├── vite.config.ts
        ├── tsconfig.app.json
        └── src/
            ├── App.tsx           # Root component & hash-based router
            ├── api.ts            # Typed fetch helpers for the FastAPI backend
            ├── index.css         # Global design tokens & styles
            ├── components/       # Reusable UI components (Sidebar, KpiCard, …)
            └── pages/            # Top-level page components
                ├── Dashboard.tsx
                ├── Forecasting.tsx
                ├── RiskAnalysis.tsx
                ├── RestaurantExplorer.tsx
                └── ModelPerformance.tsx


## Tech Stack

### Backend
| Python 3.10+ | Core language |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Baseline models |
| LightGBM / XGBoost | Gradient boosting models |
| Pydantic v2 | Request validation & serialisation |

### Frontend
| Tool | Purpose |
|---|---|
| React 19 | UI library |
| TypeScript | Type-safe JavaScript |
| Vite | Dev server & bundler |
| Recharts | Charts & data visualisation |
| Lucide React | Icon library |

## Quick Start

### 1 — Backend Setup

```bash
# Navigate to the project directory
cd minorProj

# Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

### 2 — Frontend Setup

```bash
# Navigate to the frontend directory
cd minorProj/frontend

# Install Node dependencies
npm install

# Start the Vite dev server
npm run dev
```
