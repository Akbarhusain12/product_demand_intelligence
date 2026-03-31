# � Product Demand Intelligence

> **XGBoost-powered demand forecasting and restock prediction system.**  
> Predicts when each customer will next purchase each product — aggregates into a live restock action board.

---

## 📸 Project Overview

Product Demand Intelligence is an end-to-end machine learning pipeline built on the **Instacart Market Basket Analysis** dataset. It learns each customer's personal repurchase rhythm for products and projects future demand onto a real calendar — then compares that against inventory to surface exactly what needs to be ordered, and when.

```
Purchase History  →  XGBoost Model  →  Demand Forecast  →  Restock Action Board
```

**Built for:** Quick commerce companies who need to stay ahead of stockouts and optimize inventory.

---

## 🏗️ Architecture

```
product-demand-intelligence/
├── app.py               # FastAPI REST API — serves dashboard + JSON endpoints
├── model_data.py        # Full 11-step ML pipeline (XGBoost integrated)
├── index.html           # Live dashboard — Chart.js, sortable table, model metrics
├── data_gen.ipynb       # Jupyter notebook — step-by-step walkthrough with plots
├── data/                # Instacart CSV files (add these before running)
│   ├── products.csv
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── departments.csv
│   └── aisles.csv
└── README.md
```

---

## ⚙️ Pipeline — 11 Steps

| Step | Function | What it does |
|:----:|---|---|
| 01 | `load_data()` | Load 6 Instacart CSV files into Pandas DataFrames |
| 02 | `get_baby_product_ids()` | Filter to department 18 (Baby) — 1,120 of 49,688 products |
| 03 | `build_user_timeline()` | Convert relative order gaps into absolute day numbers per user |
| 04 | `get_baby_orders()` | Filter 32M order rows to baby products + join with timeline |
| 05 | `compute_purchase_intervals()` | Measure days between consecutive purchases per (user, product) |
| 06 | `engineer_features()` | Build 16 features describing each buying habit |
| 07 | `train_xgboost_model()` | Train XGBoost regressor to predict next repurchase interval |
| 08 | `build_prediction_engine()` | Predict next purchase day for every (user, product) pair |
| 09 | `convert_to_calendar_dates()` | Map relative days → real calendar dates |
| 10 | `aggregate_demand_predictions()` | Count predicted buyers per product per date → demand forecast |
| 11 | `generate_restock_action_board()` | Compare demand vs inventory → prioritised order list |

---

## 🤖 XGBoost Model

### Why XGBoost over a simple mean interval?

Baby product repurchase is **habitual but not perfectly regular**. A mean interval treats every gap equally. XGBoost learns that:
- The **minimum gap** ever seen sets a hard floor (22.1% importance)
- The **median** is more robust than the mean for irregular buyers (18.1%)
- **Purchase count** determines how reliable the pattern is (13.1%)

### Feature Engineering — 16 Features

**User-Product level** — how *this customer* buys *this specific product:*

| Feature | Description |
|---|---|
| `interval_mean` | Average days between purchases |
| `interval_std` | Standard deviation of gaps |
| `interval_median` | Median gap (robust to outliers) |
| `interval_min` | Shortest gap ever seen |
| `interval_max` | Longest gap ever seen |
| `interval_cv` | Coefficient of variation (0 = perfectly regular) |
| `purchase_count` | Total times bought |
| `interval_count` | Number of intervals observed |
| `last_interval` | Most recent gap (recency signal) |
| `trend` | last_interval − mean (buying faster or slower?) |

**User level** — general buying behaviour:

| Feature | Description |
|---|---|
| `user_order_count` | Total orders placed |
| `user_product_variety` | Distinct products purchased |

**Product level** — how this product is bought across all customers:

| Feature | Description |
|---|---|
| `product_buyer_count` | Unique buyers of this product |
| `product_global_mean` | Mean interval across all buyers |
| `product_global_std` | Std of interval across all buyers |
| `product_global_median` | Median interval across all buyers |

> **Zero data leakage:** all features use `.expanding().shift(1)` — only past information is used to predict future intervals.

### Train / Test Split

Uses `GroupShuffleSplit` — all rows for a given user land in either **train or test, never both**. This prevents leakage and mirrors real deployment where the model predicts for unseen users.

```
80% train  |  20% test  |  Split by: user_id
```

### Model Performance

| Metric | Value |
|---|---|
| MAE | **15.46 days** |
| RMSE | **25.7 days** |
| R² Score | **0.1033** |
| Within 7 days | **33.5%** |
| Within 14 days | **63.7%** |
| Best iteration | 284 |

> MAE of 15.46 days means predictions are accurate enough to trigger restock orders **weeks before a customer runs out.**

### Hyperparameters

```python
xgb.XGBRegressor(
    n_estimators          = 400,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    reg_alpha             = 0.1,   # L1 regularisation
    reg_lambda            = 1.0,   # L2 regularisation
    early_stopping_rounds = 20,
    eval_metric           = "mae",
)
```

---

## 🌐 FastAPI Server

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the live dashboard |
| `GET` | `/api/health` | Liveness check + model metrics |
| `GET` | `/api/predictions` | Full demand forecast (all products, all dates) |
| `GET` | `/api/predictions?limit=N` | Top N forecast rows |
| `GET` | `/api/restock` | Restock action board sorted by urgency |
| `GET` | `/api/restock?limit=N` | Top N most critical SKUs |
| `GET` | `/api/model-metrics` | Detailed XGBoost evaluation metrics |
| `POST` | `/api/restock/custom-inventory` | Restock plan with real inventory data |

### Custom Inventory Example

```bash
curl -X POST http://127.0.0.1:5000/api/restock/custom-inventory \
  -H "Content-Type: application/json" \
  -d '{
    "inventory": [
      {"product_id": 38984, "current_stock": 5},
      {"product_id": 22067, "current_stock": 12}
    ]
  }'
```

---

## 📊 Dashboard Features

- **Model Metrics Banner** — live MAE, RMSE, R², within-7-days accuracy pulled from the API
- **5 KPI Cards** — Baby SKUs tracked, critical restock count, forecast rows, avg order qty, forecast horizon
- **Demand Timeline Chart** — bar chart filterable to 30 / 60 / 90 / All days
- **Restock Action Board** — sortable by any column, searchable by product ID, filterable by urgency level
- **Due in Next 7 Days** — top 6 products whose restock date falls within one week
- **Urgency Breakdown** — Critical / High / Medium / Low counts with colour-coded pills
- **CSV Export** — one-click download of the full restock list

---

## 🚀 Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Get the Instacart Market Basket Analysis dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset).

Required files:
```
products.csv
orders.csv
order_products__prior.csv
order_products__train.csv
departments.csv
aisles.csv
```

### 3. Set up the data folder

Create a `data/` directory in the project root and place all 6 CSV files inside:

```
project-demand-intelligence/
├── app.py
├── model_data.py
├── index.html
├── data/
│   ├── products.csv
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── departments.csv
│   └── aisles.csv
└── README.md
```

The app automatically resolves the path — no code edits needed.

### 4. Run

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```

Open → `http://127.0.0.1:5000`

> **Note:** The first request triggers the full pipeline — XGBoost training on 32M rows takes a few minutes. Results are cached for the session. Subsequent requests are instant.

---

## 📓 Notebook

`data_gen.ipynb` walks through all 14 steps with:

- Markdown explanations for every step
- Intermediate DataFrame previews
- Distribution plots for intervals and predictions
- XGBoost training with evaluation charts (predicted vs actual, error distribution, feature importance)
- Final restock board with urgency pie chart and summary printout

Run cells top-to-bottom. Steps 1–5 must complete before Steps 6–14.

---

## 📦 Dataset

**Instacart Market Basket Analysis** (Kaggle, 2017)

| File | Rows | Description |
|---|---|---|
| `products.csv` | 49,688 | Product names, departments, aisles |
| `orders.csv` | 3,421,083 | Order history with timing |
| `order_products__prior.csv` | 32,434,489 | Products in each prior order |
| `order_products__train.csv` | 1,384,617 | Training set order-product pairs |
| `departments.csv` | 21 | Department ID to name mapping |
| `aisles.csv` | 134 | Aisle ID to name mapping |

**Baby products** = department_id 18 → **1,120 SKUs** out of 49,688 total.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data processing | Python + Pandas | Industry standard for tabular data at scale |
| ML model | XGBoost | Best performance on tabular regression; built-in regularisation |
| Train/test split | scikit-learn `GroupShuffleSplit` | Prevents user-level data leakage |
| Web framework | FastAPI | High-performance async Python web framework |
| CORS | FastAPI middleware | Built-in support for cross-origin requests |
| Charting | Chart.js | Zero-build-step CDN charting |
| Fonts | DM Sans + DM Mono | Clean, readable; monospace for data values |
| Frontend | Vanilla HTML/CSS/JS | No framework; served directly by FastAPI |
| Notebook | Jupyter | Step-by-step exploration and validation |

---

## 🗺️ Restock Output Schema

```python
critical_restock_list.columns:
    product_id          # Baby product SKU
    next_restock_date   # Earliest date stock is needed
    total_demand_units  # Total forecasted demand across all upcoming dates
    current_stock       # Current inventory level
    units_to_order      # Shortfall: max(0, total_demand - current_stock)
    date_count          # Number of separate forecast dates for this product
```

**Urgency levels:**

| Level | Condition | Action |
|---|---|---|
| 🔴 Critical | units_to_order > 20 | Order immediately |
| 🟠 High | 11 – 20 units | Order within 48 hours |
| 🔵 Medium | 5 – 10 units | Order this week |
| 🟢 Low | 1 – 4 units | Schedule for next cycle |

---

## 🔮 Production Roadmap

- [ ] Replace simulated inventory with real database (PostgreSQL / BigQuery)
- [ ] Schedule pipeline nightly via Airflow or cron
- [ ] Add Optuna hyperparameter tuning
- [ ] Replace median interval with XGBoost for cold-start users (< 2 purchases)
- [ ] Add SHAP values for per-prediction explainability
- [ ] Containerise with Docker + deploy on AWS ECS or GCP Cloud Run
- [ ] Add push notifications when a product crosses the Critical threshold

---

## 👤 Author

Built as a final year MCA project — specifically designed to solve real demand forecasting challenges in quick commerce.

---

## 📄 License

MIT License — free to use, modify, and build on.
