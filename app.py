from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_data import run_full_pipeline, generate_restock_action_board
import pandas as pd
from typing import Optional, List
import os
import threading

app = FastAPI(title="Product Demand Intelligence", version="1.0")

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Relative path resolves correctly on any machine (Windows, Mac, Linux) ───
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Session cache — pipeline runs once on first request
# ---------------------------------------------------------------------------

_cache = {}
_cache_lock = threading.Lock()  # The bouncer for our cache


def get_pipeline_results():
    """Run the full pipeline once and cache results. Uses a Lock to prevent race conditions."""
    with _cache_lock:  # Forces concurrent API requests to wait in line
        if "results" not in _cache:
            print("\n>>> CACHE MISS: Booting the ML Pipeline (This will only happen once) <<<\n")
            _cache["results"] = run_full_pipeline(DATA_DIR)
        else:
            print("\n>>> CACHE HIT: Serving from memory <<<\n")

    return _cache["results"]


def serialize_df(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-serializable list of dicts."""
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Pydantic Models for Request Bodies
# ---------------------------------------------------------------------------


class InventoryItem(BaseModel):
    product_id: int
    current_stock: int


class InventoryRequest(BaseModel):
    inventory: List[InventoryItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    """Liveness check. Returns XGBoost model metrics when pipeline is ready."""
    try:
        results = get_pipeline_results()
        return {
            "status": "ok",
            "model": "XGBoost",
            "metrics": results.get("eval_metrics", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
def get_predictions(limit: Optional[int] = Query(None)):
    """
    Return the full XGBoost demand forecast.
    Optional query param: ?limit=N
    """
    try:
        results = get_pipeline_results()
        df = results["final_predictions"]

        if limit:
            df = df.head(limit)

        return {
            "stats": results["stats"],
            "predictions": serialize_df(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/restock")
def get_restock(limit: Optional[int] = Query(None)):
    """
    Return the XGBoost-powered restock action board.
    Optional query param: ?limit=N
    """
    try:
        results = get_pipeline_results()
        df = results["critical_restock"]

        if limit:
            df = df.head(limit)

        return {
            "stats": results["stats"],
            "restock_list": serialize_df(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/restock/custom-inventory")
def restock_with_custom_inventory(body: InventoryRequest):
    """
    Accept real inventory data and return a restock plan.

    Request body (JSON):
    {
        "inventory": [
            {"product_id": 123, "current_stock": 5},
            ...
        ]
    }
    """
    try:
        inventory_data = [item.dict() for item in body.inventory]
        inventory_df = pd.DataFrame(inventory_data)

        if not {"product_id", "current_stock"}.issubset(inventory_df.columns):
            raise HTTPException(
                status_code=400,
                detail="Each item must have 'product_id' and 'current_stock'",
            )

        results = get_pipeline_results()
        critical = generate_restock_action_board(
            final_predictions=results["final_predictions"],
            baby_product_ids=None,
            inventory_df=inventory_df,
        )

        return {
            "total_critical_skus": int(len(critical)),
            "restock_list": serialize_df(critical),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-metrics")
def model_metrics():
    """Return detailed XGBoost training and evaluation metrics."""
    try:
        results = get_pipeline_results()
        return {
            "model": "XGBoost",
            "metrics": results.get("eval_metrics", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def index():
    """Serve the frontend dashboard."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)