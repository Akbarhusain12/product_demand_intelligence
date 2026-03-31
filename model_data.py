"""
model_data.py
----------------
End-to-end Product Demand Prediction pipeline using XGBoost.

Pipeline steps:
    1.  load_data(data_dir)
    2.  get_baby_product_ids(product_df)
    3.  build_user_timeline(orders_df)
    4.  get_baby_orders(order_products_prior_df, baby_product_ids, prior_orders)
    5.  compute_purchase_intervals(baby_orders)
    6.  engineer_features(baby_orders)
    7.  train_xgboost_model(feature_df)
    8.  build_prediction_engine(model, baby_orders, feature_df)
    9.  convert_to_calendar_dates(prediction_engine, baby_orders)
    10. aggregate_demand_predictions(prediction_engine)
    11. generate_restock_action_board(final_predictions, baby_product_ids)

Install:
    pip install pandas numpy xgboost scikit-learn
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

warnings.filterwarnings("ignore")

# Features used by XGBoost — order must stay consistent
FEATURE_COLS = [
    "interval_mean",
    "interval_std",
    "interval_median",
    "interval_min",
    "interval_max",
    "interval_cv",
    "purchase_count",
    "interval_count",
    "last_interval",
    "trend",
    "user_order_count",
    "user_product_variety",
    "product_buyer_count",
    "product_global_mean",
    "product_global_std",
    "product_global_median",
]


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> dict:
    """
    Load all required Instacart CSV files from the given directory.

    Parameters
    ----------
    data_dir : str  —  path to the folder containing the CSV files.

    Returns
    -------
    dict with keys:
        'products', 'orders', 'departments',
        'order_products_prior', 'order_products_train', 'aisles'
    """
    return {
        "products":             pd.read_csv(os.path.join(data_dir, "products.csv")),
        "orders":               pd.read_csv(os.path.join(data_dir, "orders.csv")),
        "departments":          pd.read_csv(os.path.join(data_dir, "departments.csv")),
        "order_products_prior": pd.read_csv(os.path.join(data_dir, "order_products__prior.csv")),
        "order_products_train": pd.read_csv(os.path.join(data_dir, "order_products__train.csv")),
        "aisles":               pd.read_csv(os.path.join(data_dir, "aisles.csv")),
    }


# ---------------------------------------------------------------------------
# 2. Filter Baby Products  (department_id = 18)
# ---------------------------------------------------------------------------

def get_baby_product_ids(product_df: pd.DataFrame, department_id: int = 18) -> pd.Series:
    """
    Return a Series of product_ids belonging to the baby department.

    Parameters
    ----------
    product_df    : Products DataFrame with columns [product_id, department_id, ...]
    department_id : Baby department ID (default: 18)
    """
    return product_df.loc[
        product_df["department_id"] == department_id,
        "product_id",
    ]


# ---------------------------------------------------------------------------
# 3. Build User Purchase Timeline
# ---------------------------------------------------------------------------

def build_user_timeline(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to prior orders and compute a cumulative day counter per user
    so each order has an absolute 'day number' on that user's timeline.

    Parameters
    ----------
    orders_df : Full orders DataFrame

    Returns
    -------
    DataFrame of prior orders with extra column 'user_timeline_day'
    """
    prior_orders = orders_df[orders_df["eval_set"] == "prior"].copy()
    prior_orders = prior_orders.sort_values(["user_id", "order_number"])
    prior_orders["days_since_prior_order"] = (
        prior_orders["days_since_prior_order"].fillna(0)
    )
    prior_orders["user_timeline_day"] = (
        prior_orders.groupby("user_id")["days_since_prior_order"].cumsum()
    )
    return prior_orders


# ---------------------------------------------------------------------------
# 4. Get Baby Orders (with timeline info)
# ---------------------------------------------------------------------------

def get_baby_orders(
    order_products_prior_df: pd.DataFrame,
    baby_product_ids: pd.Series,
    prior_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter prior order-product records to baby products only, then enrich
    with user timeline data (user_id, order_number, user_timeline_day).

    Parameters
    ----------
    order_products_prior_df : Prior order-product mapping DataFrame
    baby_product_ids        : Series of baby product IDs
    prior_orders            : Output of build_user_timeline()
    """
    baby_order_products = order_products_prior_df[
        order_products_prior_df["product_id"].isin(baby_product_ids)
    ]
    return baby_order_products.merge(
        prior_orders[[
            "order_id", "user_id", "order_number",
            "user_timeline_day", "days_since_prior_order",
        ]],
        on="order_id",
        how="inner",
    )


# ---------------------------------------------------------------------------
# 5. Compute Purchase Intervals
# ---------------------------------------------------------------------------

def compute_purchase_intervals(baby_orders: pd.DataFrame) -> pd.DataFrame:
    """
    For each (user, product) pair, compute days between consecutive purchases.

    Adds column 'true_days_between_purchase'  (NaN for the first purchase).

    Parameters
    ----------
    baby_orders : Output of get_baby_orders()
    """
    baby_orders = baby_orders.sort_values(["user_id", "product_id", "order_number"])
    baby_orders["true_days_between_purchase"] = (
        baby_orders
        .groupby(["user_id", "product_id"])["user_timeline_day"]
        .diff()
    )
    return baby_orders


# ---------------------------------------------------------------------------
# 6. Engineer Features
# ---------------------------------------------------------------------------

def engineer_features(baby_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Build the XGBoost feature matrix from purchase history.

    Each row = one purchase interval observation for a (user_id, product_id) pair.
    All features use expanding().shift(1) — only past information is used,
    so there is zero data leakage.

    Features engineered
    -------------------
    User-product level:
        interval_mean, interval_std, interval_median, interval_min, interval_max
        interval_cv       — coefficient of variation  (0 = perfectly regular)
        purchase_count    — total purchases of this product by this user
        interval_count    — number of observed intervals
        last_interval     — most recent gap  (recency signal)
        trend             — last_interval − interval_mean  (buying faster/slower?)

    User level:
        user_order_count     — total orders this user has placed
        user_product_variety — distinct products this user has bought

    Product level:
        product_buyer_count   — unique buyers of this product
        product_global_mean   — mean interval across all buyers
        product_global_std    — std of interval across all buyers
        product_global_median — median interval across all buyers

    Parameters
    ----------
    baby_orders : Output of compute_purchase_intervals()

    Returns
    -------
    DataFrame with columns: user_id, product_id, order_number,
                             [FEATURE_COLS], target
    """
    df = baby_orders.dropna(subset=["true_days_between_purchase"]).copy()
    df = df.sort_values(["user_id", "product_id", "order_number"])

    # ── Expanding user-product features (no leakage) ─────────────────────────
    grp = df.groupby(["user_id", "product_id"])["true_days_between_purchase"]

    df["interval_mean"]   = grp.transform(lambda x: x.expanding().mean().shift(1))
    df["interval_std"]    = grp.transform(lambda x: x.expanding().std().shift(1).fillna(0))
    df["interval_median"] = grp.transform(lambda x: x.expanding().median().shift(1))
    df["interval_min"]    = grp.transform(lambda x: x.expanding().min().shift(1))
    df["interval_max"]    = grp.transform(lambda x: x.expanding().max().shift(1))
    df["interval_count"]  = grp.transform(lambda x: x.expanding().count().shift(1).fillna(0))
    df["last_interval"]   = grp.transform(lambda x: x.shift(1))

    df["interval_cv"] = np.where(
        df["interval_mean"] > 0,
        df["interval_std"] / df["interval_mean"],
        0,
    )
    df["purchase_count"] = df["interval_count"] + 1
    df["trend"]          = df["last_interval"] - df["interval_mean"]

    # ── User-level features ───────────────────────────────────────────────────
    user_stats = (
        baby_orders
        .groupby("user_id")
        .agg(
            user_order_count    =("order_number", "nunique"),
            user_product_variety=("product_id",   "nunique"),
        )
        .reset_index()
    )
    df = df.merge(user_stats, on="user_id", how="left")

    # ── Product-level features ────────────────────────────────────────────────
    product_stats = (
        df.groupby("product_id")["true_days_between_purchase"]
        .agg(
            product_buyer_count  ="count",
            product_global_mean  ="mean",
            product_global_std   ="std",
            product_global_median="median",
        )
        .reset_index()
    )
    product_stats["product_global_std"] = product_stats["product_global_std"].fillna(0)
    df = df.merge(product_stats, on="product_id", how="left")

    # Drop rows where expanding features are still NaN
    df = df.dropna(subset=FEATURE_COLS)

    df["target"] = df["true_days_between_purchase"]
    return df[["user_id", "product_id", "order_number"] + FEATURE_COLS + ["target"]]


# ---------------------------------------------------------------------------
# 7. Train XGBoost Model
# ---------------------------------------------------------------------------

def train_xgboost_model(
    feature_df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple:
    """
    Train an XGBoost regressor to predict days_between_purchase.

    Uses GroupShuffleSplit to split by user — no user appears in both
    train and test, preventing leakage and simulating real deployment.

    Parameters
    ----------
    feature_df : Output of engineer_features()
    test_size  : Fraction of users held out for evaluation (default 20%)

    Returns
    -------
    (model, eval_metrics, feature_df)
        model        : Fitted XGBRegressor
        eval_metrics : dict — mae, rmse, r2, within_7_days_pct,
                               within_14_days_pct, n_train, n_test, best_iteration
        feature_df   : Same input DataFrame (for pipeline chaining)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(feature_df, groups=feature_df["user_id"]))

    X    = feature_df[FEATURE_COLS]
    y    = feature_df["target"]
    X_tr = X.iloc[train_idx];  y_tr = y.iloc[train_idx]
    X_te = X.iloc[test_idx];   y_te = y.iloc[test_idx]

    model = xgb.XGBRegressor(
        n_estimators         = 400,
        max_depth            = 6,
        learning_rate        = 0.05,
        subsample            = 0.8,
        colsample_bytree     = 0.8,
        min_child_weight     = 5,
        reg_alpha            = 0.1,
        reg_lambda           = 1.0,
        objective            = "reg:squarederror",
        random_state         = 42,
        n_jobs               = -1,
        early_stopping_rounds= 20,
        eval_metric          = "mae",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    preds  = np.clip(model.predict(X_te), 1, None)
    errors = np.abs(preds - y_te.values)

    eval_metrics = {
        "mae":                round(float(mean_absolute_error(y_te, preds)), 2),
        "rmse":               round(float(np.sqrt(mean_squared_error(y_te, preds))), 2),
        "r2":                 round(float(r2_score(y_te, preds)), 4),
        "within_7_days_pct":  round(float((errors <= 7).mean() * 100), 1),
        "within_14_days_pct": round(float((errors <= 14).mean() * 100), 1),
        "n_train":            int(len(X_tr)),
        "n_test":             int(len(X_te)),
        "best_iteration":     int(model.best_iteration),
    }

    return model, eval_metrics, feature_df


# ---------------------------------------------------------------------------
# 8. Build Prediction Engine
# ---------------------------------------------------------------------------

def build_prediction_engine(
    model: xgb.XGBRegressor,
    baby_orders: pd.DataFrame,
    feature_df: pd.DataFrame,
    min_intervals: int = 2,
) -> pd.DataFrame:
    """
    Use the trained XGBoost model to predict the next purchase day for every
    (user, product) pair.

    Uses the LATEST feature snapshot per pair (most recent interval row) to
    make the forward prediction.

    Parameters
    ----------
    model         : Fitted XGBRegressor from train_xgboost_model()
    baby_orders   : Output of compute_purchase_intervals()
    feature_df    : Output of engineer_features()
    min_intervals : Minimum interval_count required to include a habit

    Returns
    -------
    DataFrame with columns:
        user_id, product_id, last_purchase_day, predicted_next_purchase_day
    """
    # Latest feature snapshot per (user, product)
    latest = (
        feature_df
        .sort_values(["user_id", "product_id", "order_number"])
        .groupby(["user_id", "product_id"])
        .last()
        .reset_index()
    )
    latest = latest[latest["interval_count"] >= min_intervals].copy()

    # Predict interval with XGBoost
    latest["predicted_interval"] = np.clip(
        model.predict(latest[FEATURE_COLS]), 1, None
    )

    # Get last purchase day per (user, product)
    last_purchases = (
        baby_orders
        .groupby(["user_id", "product_id"])["user_timeline_day"]
        .max()
        .reset_index()
        .rename(columns={"user_timeline_day": "last_purchase_day"})
    )

    engine = latest.merge(last_purchases, on=["user_id", "product_id"], how="inner")
    engine["predicted_next_purchase_day"] = (
        engine["last_purchase_day"] + engine["predicted_interval"]
    )

    return engine[["user_id", "product_id", "last_purchase_day",
                   "predicted_next_purchase_day"]]


# ---------------------------------------------------------------------------
# 9. Convert Timeline Days → Calendar Dates
# ---------------------------------------------------------------------------

def convert_to_calendar_dates(
    prediction_engine: pd.DataFrame,
    baby_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map relative day numbers to real calendar dates.

    The highest observed timeline day is anchored to today's actual date.
    All predicted days are offset from this anchor.

    Parameters
    ----------
    prediction_engine : Output of build_prediction_engine()
    baby_orders       : Output of compute_purchase_intervals()

    Returns
    -------
    prediction_engine with extra columns:
        'days_from_today'         – relative offset from today
        'predicted_calendar_date' – real-world calendar date
    """
    today          = pd.Timestamp.today()
    global_max_day = baby_orders["user_timeline_day"].max()

    prediction_engine = prediction_engine.copy()
    prediction_engine["days_from_today"] = (
        prediction_engine["predicted_next_purchase_day"] - global_max_day
    )
    prediction_engine["predicted_calendar_date"] = (
        today + pd.to_timedelta(prediction_engine["days_from_today"], unit="D")
    )
    
    prediction_engine = prediction_engine[prediction_engine['predicted_calendar_date'] >= today].copy()
    
    return prediction_engine


# ---------------------------------------------------------------------------
# 10. Aggregate Demand Predictions
# ---------------------------------------------------------------------------

def aggregate_demand_predictions(prediction_engine: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up individual user predictions into a daily demand forecast per product.

    Parameters
    ----------
    prediction_engine : Output of convert_to_calendar_dates()

    Returns
    -------
    DataFrame with columns:
        predicted_calendar_date, product_id, predicted_demand_units
    """
    return (
        prediction_engine[["user_id", "product_id", "predicted_calendar_date"]]
        .groupby(["predicted_calendar_date", "product_id"])
        .agg(predicted_demand_units=("user_id", "count"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# 11. Generate Restock Action Board
# ---------------------------------------------------------------------------

def generate_restock_action_board(
    final_predictions: pd.DataFrame,
    baby_product_ids,
    inventory_df: pd.DataFrame = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compare predicted demand against inventory. Flag and quantify shortfalls.

    Aggregates per product_id across all forecast dates so each product
    appears exactly once in the output.

    Parameters
    ----------
    final_predictions : Output of aggregate_demand_predictions()
    baby_product_ids  : Series of baby product IDs (for inventory simulation)
    inventory_df      : Optional DataFrame [product_id, current_stock].
                        Pass real stock data here in production.
    random_seed       : Seed for reproducible simulation (default: 42)

    Returns
    -------
    DataFrame sorted by units_to_order descending with columns:
        product_id, next_restock_date, total_demand_units,
        current_stock, units_to_order, date_count
    """
    if inventory_df is None:
        np.random.seed(random_seed)
        inventory_df = pd.DataFrame({
            "product_id":    baby_product_ids,
            "current_stock": np.random.randint(0, 40, size=len(baby_product_ids)),
        })

    action_board = final_predictions.merge(inventory_df, on="product_id", how="left")
    action_board["current_stock"] = action_board["current_stock"].fillna(0)

    aggregated = (
        action_board
        .groupby("product_id", as_index=False)
        .agg(
            total_demand_units =("predicted_demand_units",  "sum"),
            current_stock      =("current_stock",           "min"),
            next_restock_date  =("predicted_calendar_date", "min"),
            date_count         =("predicted_calendar_date", "count"),
        )
    )

    aggregated["units_to_order"] = np.where(
        aggregated["total_demand_units"] > aggregated["current_stock"],
        aggregated["total_demand_units"] - aggregated["current_stock"],
        0,
    )

    critical = (
        aggregated[aggregated["units_to_order"] > 0]
        .copy()
        .sort_values("units_to_order", ascending=False)
        .reset_index(drop=True)
    )

    return critical[[
        "product_id", "next_restock_date", "total_demand_units",
        "current_stock", "units_to_order", "date_count",
    ]]


# ---------------------------------------------------------------------------
# Master Pipeline — single entry-point for FastAPI
# ---------------------------------------------------------------------------

def run_full_pipeline(data_dir: str) -> dict:
    """
    Run the complete XGBoost demand prediction and restock pipeline end-to-end.

    Parameters
    ----------
    data_dir : Path to the folder containing the Instacart CSV files

    Returns
    -------
    dict with keys:
        'final_predictions' – pd.DataFrame
        'critical_restock'  – pd.DataFrame
        'stats'             – dict  (for API responses)
        'model'             – fitted XGBRegressor
        'eval_metrics'      – dict  (MAE, RMSE, R², accuracy %)
    """
    print("Step 1  — Loading data...")
    data = load_data(data_dir)

    print("Step 2  — Filtering baby products (dept 18)...")
    baby_product_ids = get_baby_product_ids(data["products"])

    print("Step 3  — Building user timelines...")
    prior_orders = build_user_timeline(data["orders"])

    print("Step 4  — Getting baby orders...")
    baby_orders = get_baby_orders(
        data["order_products_prior"], baby_product_ids, prior_orders
    )  

    print("Step 5  — Computing purchase intervals...")
    baby_orders = compute_purchase_intervals(baby_orders)

    print("Step 6  — Engineering features...")
    feature_df = engineer_features(baby_orders)
    print(f"          {feature_df.shape[0]:,} rows × {len(FEATURE_COLS)} features")

    print("Step 7  — Training XGBoost...")
    model, eval_metrics, feature_df = train_xgboost_model(feature_df)
    print(f"          MAE {eval_metrics['mae']}d  |  RMSE {eval_metrics['rmse']}d  |  R² {eval_metrics['r2']}")

    print("Step 8  — Building prediction engine...")
    prediction_engine = build_prediction_engine(model, baby_orders, feature_df)

    print("Step 9  — Converting to calendar dates...")
    prediction_engine = convert_to_calendar_dates(prediction_engine, baby_orders)

    print("Step 10 — Aggregating demand predictions...")
    final_predictions = aggregate_demand_predictions(prediction_engine)

    print("Step 11 — Generating restock action board...")
    critical_restock = generate_restock_action_board(final_predictions, baby_product_ids)

    print(f"\nDone. {len(critical_restock):,} critical SKUs flagged.")

    return {
        "final_predictions": final_predictions,
        "critical_restock":  critical_restock,
        "stats": {
            "total_baby_products":      int(len(baby_product_ids)),
            "total_demand_rows":        int(len(final_predictions)),
            "critical_skus_to_restock": int(len(critical_restock)),
            "model_mae":                eval_metrics["mae"],
            "model_r2":                 eval_metrics["r2"],
        },
        "model":        model,
        "eval_metrics": eval_metrics,
    }