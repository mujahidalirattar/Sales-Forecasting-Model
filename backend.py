"""
backend.py
==========
Frozen model, data pipeline, and business logic for the Walmart Sales
Forecasting dashboard.

This module is intentionally free of any web framework. It exposes the exact
same computations the finished project shipped with: the 17 feature contract,
the train + features + stores merge, the KPI aggregation, the single row feature
builder, the model prediction alignment, and the inventory and staffing engines.

The only change from the original Streamlit build is infrastructural: Streamlit
cache decorators are replaced with functools.lru_cache, and joblib is imported
lazily inside load_model so this module can be imported and unit tested with only
pandas and numpy present. No calculation, threshold, or feature order changes.

Model contract (strict 17 feature vector, exact order):
  [Store, Dept, Type, Size, IsHoliday, Temperature, Fuel_Price,
   MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5,
   CPI, Unemployment, Year, Month, Week]
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd


# ============================================================================= #
# CONSTANTS AND PATHS (unchanged model/data contract)
# ============================================================================= #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_ORDER = [
    "Store", "Dept", "Type", "Size", "IsHoliday", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "Year", "Month", "Week",
]
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
TARGET_COL = "Weekly_Sales"

TYPE_TO_CODE = {"A": 1, "B": 2, "C": 3}
CODE_TO_TYPE = {1: "A", 2: "B", 3: "C"}
TYPE_LABELS = {
    1: "Type A : Supercenter",
    2: "Type B : Discount Store",
    3: "Type C : Neighborhood Market",
}


def resolve_model_path():
    """Locate the model file, preferring the Model/ subfolder next to this script."""
    candidates = [
        os.path.join(BASE_DIR, "Model", "walmart_rf_model.pkl"),
        os.path.join(BASE_DIR, "walmart_rf_model.pkl"),
        os.path.join(os.getcwd(), "walmart_rf_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(BASE_DIR, "Model", "walmart_rf_model.pkl")


def resolve_data_path(filename):
    """Locate a data CSV next to this script, then in the current working directory."""
    for path in (os.path.join(BASE_DIR, filename), os.path.join(os.getcwd(), filename)):
        if os.path.exists(path):
            return path
    return os.path.join(BASE_DIR, filename)


MODEL_PATH = resolve_model_path()


# ============================================================================= #
# MODEL + DATA PIPELINE (cached) - logic preserved from the finished backend
# ============================================================================= #

@lru_cache(maxsize=1)
def load_model():
    """Load the trained RandomForest model. Returns (model, error_message)."""
    try:
        import joblib
        if not os.path.exists(MODEL_PATH):
            return None, f"Model file not found at: {MODEL_PATH}"
        model = joblib.load(MODEL_PATH, mmap_mode='r')
        return model, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Failed to load model: {exc}"


@lru_cache(maxsize=1)
def load_master_data():
    """Merge train + features + stores into one analytics ready DataFrame.

    Returns (dataframe, error_message). Callers must copy before mutating, since
    the frame is cached and shared.
    """
    try:
        train = pd.read_csv(resolve_data_path("train.csv"))
        feats = pd.read_csv(resolve_data_path("features.csv"))
        stores = pd.read_csv(resolve_data_path("stores.csv"))
    except FileNotFoundError as exc:
        return None, f"Data file missing: {exc}"

    df = train.merge(stores, on="Store", how="left")
    df = df.merge(feats, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    if "IsHoliday_feat" in df.columns:
        df = df.drop(columns=["IsHoliday_feat"])
    df["TypeLetter"] = df["Type"].astype(str).str.strip().str.upper()
    df["Type"] = df["TypeLetter"].map(TYPE_TO_CODE).fillna(0).astype(int)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year.astype("int16")
    df["Month"] = df["Date"].dt.month.astype("int8")
    df["Week"] = df["Date"].dt.isocalendar().week.astype("int16")
    for col in MARKDOWN_COLS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0).astype("float32")
    df["IsHoliday"] = df["IsHoliday"].astype(bool).astype("int8")
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Weekly_Sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce").fillna(0).astype("int32")
    df["Store"] = df["Store"].astype("int16")
    df["Dept"] = df["Dept"].astype("int16")
    df = df.dropna(subset=["Weekly_Sales", "CPI", "Unemployment", "Temperature", "Fuel_Price"])
    return df, None


@lru_cache(maxsize=8)
def get_scatter_sample(type_filter_key, n=4000, seed=42):
    """Deterministic sample used for the regression scatter (keeps rendering fast)."""
    df, err = load_master_data()
    if err or df is None:
        return None
    view = df if not type_filter_key else df[df["Type"].isin(type_filter_key)]
    if len(view) > n:
        view = view.sample(n=n, random_state=seed)
    return view.reset_index(drop=True)


@lru_cache(maxsize=1)
def dataset_kpis():
    """Top level KPI strip values computed from the master dataset."""
    df, err = load_master_data()
    if err or df is None:
        return None
    return {
        "records": int(len(df)),
        "stores": int(df["Store"].nunique()),
        "depts": int(df["Dept"].nunique()),
        "avg_weekly": float(df["Weekly_Sales"].mean()),
        "total_sales": float(df["Weekly_Sales"].sum()),
        "date_min": df["Date"].min(),
        "date_max": df["Date"].max(),
    }


def build_feature_row(store, dept, store_type, size, is_holiday, temp, fuel,
                      cpi, unemp, year, month, week, markdowns=(0.0, 0.0, 0.0, 0.0, 0.0)):
    """Build a single row DataFrame in the exact 17 feature order the model expects."""
    values = {
        "Store": store, "Dept": dept, "Type": store_type, "Size": size,
        "IsHoliday": is_holiday, "Temperature": temp, "Fuel_Price": fuel,
        "MarkDown1": markdowns[0], "MarkDown2": markdowns[1], "MarkDown3": markdowns[2],
        "MarkDown4": markdowns[3], "MarkDown5": markdowns[4],
        "CPI": cpi, "Unemployment": unemp, "Year": year, "Month": month, "Week": week,
    }
    return pd.DataFrame([[values[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)


def model_predict(mdl, frame):
    """Predict after aligning columns to the model's own trained feature order.

    Guarantees correct positional mapping (and no feature name warnings) even if the
    frame's column order ever drifts from the model's expected order.
    """
    names = getattr(mdl, "feature_names_in_", None)
    if names is not None:
        aligned = frame.reindex(columns=list(names))
    else:
        aligned = frame[FEATURE_ORDER]
    return mdl.predict(aligned)


# ============================================================================= #
# PRESCRIPTIVE ENGINES (pure, unit testable) - preserved exactly
# ============================================================================= #

def money(value, decimals=0):
    return f"${value:,.{decimals}f}"


def inventory_status(current_stock, predicted_sales):
    """Classify inventory posture against forecast demand with a 20% overstock buffer."""
    current_stock = float(current_stock)
    predicted = max(0.0, float(predicted_sales))
    buffer_high = predicted * 1.20

    if current_stock < predicted:
        gap = predicted - current_stock
        return {
            "status": "STOCKOUT RISK", "level": "alert", "buffer_high": buffer_high,
            "detail": (f"On-hand inventory of {money(current_stock)} is below forecast demand "
                       f"of {money(predicted)}. Expedite replenishment of {money(gap)} to protect sales."),
            "action": f"Reorder ~ {money(gap)}",
        }
    if current_stock > buffer_high:
        excess = current_stock - buffer_high
        return {
            "status": "OVERSTOCK RISK", "level": "warning", "buffer_high": buffer_high,
            "detail": (f"On-hand inventory of {money(current_stock)} exceeds the 20% safety buffer "
                       f"({money(buffer_high)}). Roughly {money(excess)} of capital is over-committed."),
            "action": "Pause reorders / run clearance",
        }
    return {
        "status": "OPTIMAL", "level": "success", "buffer_high": buffer_high,
        "detail": (f"On-hand inventory of {money(current_stock)} sits within the healthy range up to "
                   f"the {money(buffer_high)} buffer. No corrective action required."),
        "action": "Maintain current levels",
    }


def staffing_plan(predicted_sales, is_holiday):
    """Labor model: 1 worker per $5,000 forecast sales; 35% cashiers / 65% floor."""
    predicted = max(0.0, float(predicted_sales))
    total = int(np.ceil(predicted / 5000.0)) if predicted > 0 else 0

    cashiers = int(round(total * 0.35))
    if total > 0:
        cashiers = max(1, cashiers)
    cashiers = min(cashiers, total)
    floor = total - cashiers

    surge = bool(is_holiday) or predicted > 25000.0
    if bool(is_holiday) and predicted > 25000.0:
        reason = "Holiday week and forecast above $25,000"
    elif bool(is_holiday):
        reason = "Holiday week"
    elif predicted > 25000.0:
        reason = "Forecast above $25,000"
    else:
        reason = "Standard weekday coverage is sufficient"

    return {
        "total": total, "cashiers": cashiers, "floor": floor,
        "surge": surge, "reason": reason,
    }


# ============================================================================= #
# TRAINING FRAME PREP (used by the retraining pipeline) - preserved exactly
# ============================================================================= #

def prepare_training_frame(df_raw):
    """Coerce an uploaded/reference frame into (X, y, report).

    Maps Type A/B/C -> 1/2/3, derives Year/Month/Week from Date when needed,
    defaults MarkDowns to 0, and validates the strict 17 feature contract + target.
    Returns (X, y, report_dict). Raises ValueError with a clear message if invalid.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report = {"present": [], "missing": [], "derived": []}

    if "Type" in df.columns and df["Type"].dtype == object:
        df["Type"] = (df["Type"].astype(str).str.strip().str.upper()
                      .map(TYPE_TO_CODE).fillna(df["Type"]))
        df["Type"] = pd.to_numeric(df["Type"], errors="coerce")

    if "IsHoliday" in df.columns:
        df["IsHoliday"] = (df["IsHoliday"].astype(str).str.strip().str.lower()
                           .map({"true": 1, "false": 0, "1": 1, "0": 0,
                                 "yes": 1, "no": 0}).fillna(0).astype(int))

    for col in MARKDOWN_COLS:
        if col not in df.columns:
            df[col] = 0.0
            report["derived"].append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "Date" in df.columns and not all(c in df.columns for c in ["Year", "Month", "Week"]):
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if "Year" not in df.columns:
            df["Year"] = parsed.dt.year; report["derived"].append("Year")
        if "Month" not in df.columns:
            df["Month"] = parsed.dt.month; report["derived"].append("Month")
        if "Week" not in df.columns:
            df["Week"] = parsed.dt.isocalendar().week.astype("Int64"); report["derived"].append("Week")

    required = FEATURE_ORDER + [TARGET_COL]
    for col in required:
        (report["present"] if col in df.columns else report["missing"]).append(col)
    if report["missing"]:
        raise ValueError("Missing required column(s): " + ", ".join(report["missing"]))

    for col in FEATURE_ORDER + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    clean = df.dropna(subset=FEATURE_ORDER + [TARGET_COL])
    report["rows_used"] = int(len(clean))
    report["rows_dropped"] = int(len(df) - len(clean))
    if len(clean) < 50:
        raise ValueError(f"Only {len(clean)} valid rows after cleaning; need at least 50 to train.")

    X = clean[FEATURE_ORDER].astype(float)
    y = clean[TARGET_COL].astype(float)
    return X, y, report
