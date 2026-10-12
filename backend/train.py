"""
TRAINING (run once):  python backend/train.py

Reads the Skytrax Excel file, cleans it, trains a multiple linear regression,
tests it on unseen reviews, and saves the trained model + its scores to model/.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Skytrax_Airline_Review_Data.xlsx"
MODEL_DIR = ROOT / "model"

Y_COL = "over_all_rating"
X_COLS = ["value_for_money", "ground_service", "seat_comfort", "cabin_staff_service", "food__beverages"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)
    print(f"Loaded {len(df):,} reviews")
    df = df.dropna(subset=X_COLS + [Y_COL])
    df = df.drop_duplicates(subset=["review_text", "airline_name", Y_COL]).reset_index(drop=True)
    print(f"Kept {len(df):,} complete, de-duplicated reviews")
    return df


def split(df: pd.DataFrame, seed: int = 42, train_share: float = 0.8):
    idx = np.random.default_rng(seed).permutation(len(df))
    n_train = int(train_share * len(df))
    return df.iloc[idx[:n_train]], df.iloc[idx[n_train:]]


def evaluate(model, X, y) -> dict:
    y_hat = model.predict(X)
    return {
        "MAE": round(float(mean_absolute_error(y, y_hat)), 4),
        "MSE": round(float(mean_squared_error(y, y_hat)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, y_hat))), 4),
        "R2": round(float(r2_score(y, y_hat)), 4),
    }


def main():
    df = load_and_clean()
    train, test = split(df)
    print(f"Training on {len(train):,} reviews, testing on {len(test):,}")

    model = LinearRegression().fit(train[X_COLS], train[Y_COL])   # least squares

    report = {
        "intercept": float(model.intercept_),
        "coefficients": dict(zip(X_COLS, map(float, model.coef_))),
        "train": evaluate(model, train[X_COLS], train[Y_COL]),
        "test": evaluate(model, test[X_COLS], test[Y_COL]),
        "n_train": len(train),
        "n_test": len(test),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    # Save the cleaned reviews + the model's prediction for each one.
    # The dashboard uses these to build airline reports, trends and "reviews that don't add up".
    reviews = df[["airline_name", "review_date", "review_header", "recommended", Y_COL] + X_COLS].copy()
    reviews["year"] = pd.to_datetime(reviews["review_date"]).dt.year
    reviews["predicted"] = model.predict(df[X_COLS])
    reviews.to_pickle(MODEL_DIR / "reviews.pkl")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\nRating = {:.3f} + ".format(report["intercept"]) +
          " + ".join(f"{b:.3f}·{c}" for c, b in report["coefficients"].items()))
    print("Test:", report["test"])
    print(f"\nSaved model/model.joblib, model/metrics.json and model/reviews.pkl")


if __name__ == "__main__":
    main()
