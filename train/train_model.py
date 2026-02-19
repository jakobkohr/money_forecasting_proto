import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.features import build_ml_table, get_feature_columns

DATA_PATH = "data/synthetic_revolut_transactions.csv"
MODEL_PATH = "models/daily_spend_model.joblib"
META_PATH = "models/model_meta.json"

def main():
    df = pd.read_csv(DATA_PATH)

    # Build ML table
    ml = build_ml_table(df, payday_day=1)
    feature_cols = get_feature_columns()

    X = ml[feature_cols]
    y = ml["y_next_spend"]

    # Time-based split: train on first 80%, test on last 20%
    split_idx = int(len(ml) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        max_depth=10,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Save model + metadata
    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_cols": feature_cols,
        "payday_day": 1,
        "mae_test": float(mae),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Model trained and saved.")
    print(f"Test MAE: {mae:.2f} € (lower is better)")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {META_PATH}")

if __name__ == "__main__":
    main()
