# Flight Rating Lab (Group 7)

A small machine-learning system with three parts:

| Part | File | What it does |
|---|---|---|
| Training | `backend/train.py` | Reads the Skytrax Excel file, cleans it, trains a multiple linear regression (least squares) on 80% of reviews, tests it on the other 20%, and saves `model/model.joblib`, `model/metrics.json` and `model/reviews.pkl` (cleaned reviews + the model's prediction for each) |
| Backend | `backend/app.py` | A Flask API. Loads the model and reviews, and answers the dashboard: airline reports, trends, what-if projections |
| Frontend | `frontend/index.html` | The dashboard. Pick an airline and period to see where it wins and loses rating points, whether it is improving, what to fix first, and reviews that don't add up |

## Run it (Ubuntu)

```bash
cd ~/Desktop/"Data mining"/skytrax-rating-app

# 1. one time: environment + packages
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. put the dataset in data/  (exact name)
cp ../Skytrax_Airline_Review_Data.xlsx data/

# 3. train the model
python backend/train.py

# 4. start the backend, then open http://127.0.0.1:5000 in Chrome
python backend/app.py
```

Leave the terminal open while you use the page. Stop the server with Ctrl + C.

## API

```
GET  /api/model                  -> intercept, coefficients, train/test MAE, MSE, RMSE, R2
GET  /api/airlines               -> airlines with 100+ reviews, average rating, recommend share
GET  /api/airline/<id>?from=&to= -> services vs all airlines, points won/lost, yearly trend, odd reviews
POST /api/whatif                 body: {"airline":"american-airlines","changes":{"value_for_money":1}}
                                 -> {"current":2.51,"gain":1.0,"projected":3.51}
```

Expected training output: Rating = -1.751 + 0.996*value + 0.535*ground + 0.237*seat + 0.267*staff + 0.255*food,
test R2 0.878, RMSE 1.22.
# skytrax-rating-app
