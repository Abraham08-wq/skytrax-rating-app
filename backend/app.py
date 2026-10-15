"""
BACKEND / API SERVER:  python backend/app.py   → open http://127.0.0.1:5000

Loads the trained regression model and the cleaned reviews once, then answers the dashboard:

  GET  /api/model                 coefficients + accuracy of the trained model
  GET  /api/airlines              every airline with its review count and average rating
  GET  /api/airline/<name>        full report for one airline (optional ?from=2018&to=2025)
  POST /api/whatif                projected rating if an airline improves some services
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
FRONTEND_DIR = ROOT / "frontend"
Y_COL = "over_all_rating"
X_COLS = ["value_for_money", "ground_service", "seat_comfort", "cabin_staff_service", "food__beverages",
          "inflight_entertainment", "wifi__connectivity"]
LABELS = {"value_for_money": "Value for money", "ground_service": "Ground service",
          "seat_comfort": "Seat comfort", "cabin_staff_service": "Cabin staff", "food__beverages": "Food & drinks",
          "inflight_entertainment": "Entertainment", "wifi__connectivity": "Wifi"}
MIN_REVIEWS = 100        # airlines with fewer reviews are too small to report on
MIN_PER_YEAR = 15        # years with fewer reviews are left off the trend chart

for f in ["model.joblib", "metrics.json", "reviews.pkl"]:
    if not (MODEL_DIR / f).exists():
        raise SystemExit(f"Missing model/{f}. Run  python backend/train.py  first.")

model = joblib.load(MODEL_DIR / "model.joblib")
report = json.loads((MODEL_DIR / "metrics.json").read_text())
COEF = report["coefficients"]
reviews = pd.read_pickle(MODEL_DIR / "reviews.pkl")
reviews["recommended"] = reviews["recommended"].astype(str).str.lower().eq("yes")
reviews["residual"] = reviews[Y_COL] - reviews["predicted"]

app = Flask(__name__)


def nice(name: str) -> str:
    return name.replace("-", " ").title()


def r(x, n=2):
    return None if pd.isna(x) else round(float(x), n)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/model")
def model_info():
    return jsonify(report)


@app.get("/api/airlines")
def airlines():
    g = reviews.groupby("airline_name").agg(n=(Y_COL, "size"), rating=(Y_COL, "mean"),
                                            recommend=("recommended", "mean"))
    g = g[g["n"] >= MIN_REVIEWS].sort_values("rating", ascending=False)
    years = sorted(int(y) for y in reviews["year"].unique())
    return jsonify(
        industry_rating=r(reviews[Y_COL].mean()),
        years=years,
        airlines=[{"id": a, "name": nice(a), "n": int(row.n), "rating": r(row.rating),
                   "recommend": r(row.recommend, 3)} for a, row in g.iterrows()],
    )


@app.get("/api/airline/<name>")
def airline(name):
    y_from = request.args.get("from", type=int, default=int(reviews["year"].min()))
    y_to = request.args.get("to", type=int, default=int(reviews["year"].max()))
    period = reviews[(reviews["year"] >= y_from) & (reviews["year"] <= y_to)]
    mine = period[period["airline_name"] == name]
    if len(mine) == 0:
        return jsonify(error=f"No reviews for '{name}' between {y_from} and {y_to}"), 404

    # 1. Each service: this airline vs all airlines, and what the gap is worth in rating points
    services = []
    for c in X_COLS:
        yours, industry = mine[c].mean(), period[c].mean()
        gap = yours - industry                       # stars above (+) or below (−) the average airline
        services.append({
            "id": c, "label": LABELS[c], "yours": r(yours), "industry": r(industry), "gap": r(gap),
            "per_star": r(COEF[c], 3),
            "points": r(COEF[c] * gap, 3),           # coefficient × gap = rating points won or lost
            "room": r(COEF[c] * (5 - yours), 3),     # points available if this service reached 5 stars
        })

    rating, industry_rating = mine[Y_COL].mean(), period[Y_COL].mean()
    explained = sum(s["points"] for s in services)

    # 2. Trend by year (overall + each service)
    t = mine.groupby("year").agg(n=(Y_COL, "size"), rating=(Y_COL, "mean"), **{c: (c, "mean") for c in X_COLS})
    t = t[t["n"] >= MIN_PER_YEAR]
    ind_t = period.groupby("year")[Y_COL].mean()
    trend = [{"year": int(y), "n": int(row.n), "rating": r(row.rating), "industry": r(ind_t.get(y)),
              **{c: r(row[c]) for c in X_COLS}} for y, row in t.iterrows()]

    # 3. Reviews whose overall rating does not match their service scores (biggest |actual − predicted|)
    odd = mine.reindex(mine["residual"].abs().sort_values(ascending=False).index).head(6)
    flagged = [{"header": str(row.review_header).strip('" '), "date": str(row.review_date)[:10],
                "actual": int(row[Y_COL]), "predicted": r(row.predicted, 1),
                "scores": {c: int(row[c]) for c in X_COLS}} for _, row in odd.iterrows()]

    return jsonify(
        id=name, name=nice(name), period=[y_from, y_to], n=int(len(mine)),
        rating=r(rating), industry_rating=r(industry_rating),
        recommend=r(mine["recommended"].mean(), 3),
        difference=r(rating - industry_rating), explained=r(explained),
        services=services, trend=trend, flagged=flagged,
        odd_share=r((mine["residual"].abs() >= 4).mean(), 3),
    )


@app.post("/api/whatif")
def whatif():
    """Projected rating if the airline's average score for some services rises by the given stars."""
    body = request.get_json(silent=True) or {}
    name = body.get("airline")
    mine = reviews[reviews["airline_name"] == name]
    if len(mine) == 0:
        return jsonify(error="Unknown airline"), 404
    base_scores = mine[X_COLS].mean()
    new_scores = base_scores.copy()
    for c, delta in (body.get("changes") or {}).items():
        if c in X_COLS:
            new_scores[c] = min(5.0, base_scores[c] + float(delta))
    before = float(model.predict(pd.DataFrame([base_scores]))[0])
    after = float(model.predict(pd.DataFrame([new_scores]))[0])
    current = float(mine[Y_COL].mean())
    return jsonify(current=r(current), projected=r(min(10.0, current + (after - before))),
                   gain=r(after - before), new_scores={c: r(new_scores[c]) for c in X_COLS})


if __name__ == "__main__":
    app.run(debug=False)