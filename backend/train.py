"""
TRAINING (run once):  python backend/train.py

Reads the Skytrax Excel file, cleans it, trains a multiple linear regression,
tests it on unseen reviews, and saves the trained model + its scores to model/.

Change from the previous version:
  inflight_entertainment and wifi__connectivity are now used as features.
  They are ~33% and ~64% empty, so instead of dropping those rows, the empty
  cells are filled with the MODE (most common value) of the column.
  The other 5 features still drop rows that have missing values, as before.
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

# Columns that must be complete: rows missing any of these are dropped (unchanged).
REQUIRED_COLS = ["value_for_money", "ground_service", "seat_comfort", "cabin_staff_service", "food__beverages"]

# Columns that are mostly empty: missing values are filled with the column mode.
MODE_FILL_COLS = ["inflight_entertainment", "wifi__connectivity"]

X_COLS = REQUIRED_COLS + MODE_FILL_COLS


def load_and_clean() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)
    print(f"Loaded {len(df):,} reviews")
    # Only the required columns + the target decide which rows are dropped.
    # Entertainment and wifi are NOT in this list, so their empty rows are kept.
    df = df.dropna(subset=REQUIRED_COLS + [Y_COL])
    df = df.drop_duplicates(subset=["review_text", "airline_name", Y_COL]).reset_index(drop=True)
    print(f"Kept {len(df):,} complete, de-duplicated reviews")
    return df


def split(df: pd.DataFrame, seed: int = 42, train_share: float = 0.8):
    idx = np.random.default_rng(seed).permutation(len(df))
    n_train = int(train_share * len(df))
    return df.iloc[idx[:n_train]].copy(), df.iloc[idx[n_train:]].copy()


def fill_with_mode(frame: pd.DataFrame, modes: dict) -> pd.DataFrame:
    """Replace empty cells in the mode-fill columns with the stored mode."""
    frame = frame.copy()
    for col, mode in modes.items():
        frame[col] = frame[col].fillna(mode)
    return frame


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

    # How empty were the two columns before filling?
    missing_share = {c: round(float(df[c].isna().mean()), 4) for c in MODE_FILL_COLS}
    print("Share empty before filling:", missing_share)

    train, test = split(df)

    # Mode is calculated from the TRAINING reviews only, then used for train, test and
    # the saved reviews. This keeps the test score honest (no peeking at unseen data).
    modes = {c: float(train[c].mode().iloc[0]) for c in MODE_FILL_COLS}
    print("Mode used to fill empty cells:", modes)

    train = fill_with_mode(train, modes)
    test = fill_with_mode(test, modes)
    df = fill_with_mode(df, modes)

    print(f"Training on {len(train):,} reviews, testing on {len(test):,}")

    model = LinearRegression().fit(train[X_COLS], train[Y_COL])   # least squares

    report = {
        "intercept": float(model.intercept_),
        "coefficients": dict(zip(X_COLS, map(float, model.coef_))),
        "fill_modes": modes,
        "missing_share_before_fill": missing_share,
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