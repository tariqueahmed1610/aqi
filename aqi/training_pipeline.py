import os
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
try:
    from google.colab import userdata
    HOPSWORKS_API_KEY = userdata.get("HOPSWORKS_API_KEY")
except Exception:
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise RuntimeError(
        "HOPSWORKS_API_KEY not found. Set it as a Colab Secret or env var."
    )

FG_NAME = "aqi_features"
FG_VERSION = 6  # must match the version created by feature_pipeline.py
FV_NAME = "aqi_fv"
FV_VERSION = 3  # auto-incremented here since the query changed (added city_id)

# city_id added as a feature so one model can serve all 5 cities -
# order must exactly match CITIES in feature_pipeline.py
FEATURES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "hour",
    "day_of_week", "month", "pm2_5_lag_1h", "pm2_5_lag_24h",
    "aqi_rate_of_change", "city_id",
]

# One target per forecast day
HORIZONS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZON_LABELS = {"target_aqi_24h": "24h", "target_aqi_48h": "48h", "target_aqi_72h": "72h"}


# ----------------------------------------------------------------------
# 1. LOAD DATA FROM THE FEATURE STORE (via a Feature View, once)
# ----------------------------------------------------------------------
def load_all_data():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)

    query = fg.select(FEATURES + HORIZONS + ["event_time"])

    # Still create the Feature View, so the model registry entry can be
    # linked to it for lineage - but DON'T use fv.training_data() to
    # fetch the actual data. Its X/y label-splitting has proven
    # unreliable (column names/order don't always match what you passed
    # in). query.read() returns the exact columns we asked for, by
    # name, with no hidden splitting - much safer.
    fv = fs.get_or_create_feature_view(
        name=FV_NAME,
        version=FV_VERSION,
        query=query,
        labels=HORIZONS,
    )

    df = query.read()

    # time-ordered split (80/20) - no random shuffling, this is a
    # forecasting problem so the test set must come after the train set.
    # Cities share the event_time range, so sorting all rows together
    # keeps the split a genuine "future" holdout for every city at once.
    df = df.sort_values("event_time").reset_index(drop=True)

    return mr, fv, df


# ----------------------------------------------------------------------
# 2. TRAIN + COMPARE MODELS FOR ONE HORIZON
# ----------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "MAE": round(float(mean_absolute_error(y_test, preds)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        "R2_Score": round(float(r2_score(y_test, preds)), 4),
    }
    print(f"  {name}: MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  R2={metrics['R2_Score']}")
    return metrics


def train_and_compare(X_train, X_test, y_train, y_test):
    candidates = {}

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    candidates["Ridge"] = (ridge, evaluate("Ridge", ridge, X_test, y_test))

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    candidates["RandomForest"] = (rf, evaluate("RandomForest", rf, X_test, y_test))

    xgb_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    candidates["XGBoost"] = (xgb_model, evaluate("XGBoost", xgb_model, X_test, y_test))

    best_name = min(candidates, key=lambda k: candidates[k][1]["RMSE"])
    best_model, best_metrics = candidates[best_name]
    return best_name, best_model, best_metrics


# ----------------------------------------------------------------------
# 3. SHAP EXPLANATION (best model only)
# ----------------------------------------------------------------------
def explain(best_name, best_model, X_test):
    sample = X_test.iloc[:500]
    try:
        if best_name in ("RandomForest", "XGBoost"):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(sample)
        else:
            explainer = shap.Explainer(best_model, sample)
            shap_values = explainer(sample).values

        importance = pd.DataFrame({
            "feature": FEATURES,
            "importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("importance", ascending=False)

        print("  Top 5 features (SHAP):")
        for _, row in importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
    except Exception as e:
        print(f"  SHAP explanation skipped ({e})")


# ----------------------------------------------------------------------
# 4. REGISTER BEST MODEL FOR THIS HORIZON
# ----------------------------------------------------------------------
def register_model(mr, horizon_label, best_name, best_model, best_metrics, fv):
    model_dir = f"model_dir_{horizon_label}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, f"{model_dir}/aqi_model.pkl")

    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(best_metrics, f)
    with open(f"{model_dir}/features.json", "w") as f:
        json.dump(FEATURES, f)

    aqi_model = mr.python.create_model(
        name=f"karachi_aqi_predictor_{horizon_label}",
        metrics=best_metrics,
        description=f"{best_name} model predicting AQI {horizon_label} ahead, all 5 cities (city_id feature).",
        feature_view=fv,
    )
    aqi_model.save(model_dir)
    print(f"  Registered as 'karachi_aqi_predictor_{horizon_label}' ({best_name}).")


def main():
    mr, fv, df = load_all_data()
    split_idx = int(len(df) * 0.80)

    for target_col in HORIZONS:
        horizon_label = HORIZON_LABELS[target_col]
        print(f"\n=== Horizon: {horizon_label} (target: {target_col}) ===")

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train, y_train = train_df[FEATURES], train_df[target_col]
        X_test, y_test = test_df[FEATURES], test_df[target_col]

        best_name, best_model, best_metrics = train_and_compare(
            X_train, X_test, y_train, y_test
        )
        print(f"  Best: {best_name} (RMSE={best_metrics['RMSE']})")
        explain(best_name, best_model, X_test)
        register_model(mr, horizon_label, best_name, best_model, best_metrics, fv)


if __name__ == "__main__":
    main()
