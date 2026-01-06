import pandas as pd
import numpy as np

TARGET = "ili_weighted_pct"
LAGS = [1, 2, 4, 8]

def forecast_future(df, model, region, horizon):
    # 1️⃣ Filter region & sort time
    df_r = df[df["region"] == region].copy()
    df_r = df_r.sort_values(["year", "week"]).reset_index(drop=True)

    # 2️⃣ Feature columns
    feature_cols = [f"{TARGET}_lag_{l}" for l in LAGS]

    # 3️⃣ Start from last known lags
    current_features = df_r.iloc[-1][feature_cols].to_dict()

    forecasts = []

    for _ in range(horizon):
        # 4️⃣ Create DataFrame (IMPORTANT)
        X = pd.DataFrame([current_features], columns=feature_cols)

        # 5️⃣ Predict
        pred = model.predict(X)[0]
        pred = round(float(pred), 2)
        forecasts.append(pred)

        # 6️⃣ Update lags manually (CORRECT WAY)
        current_features[f"{TARGET}_lag_8"] = current_features[f"{TARGET}_lag_4"]
        current_features[f"{TARGET}_lag_4"] = current_features[f"{TARGET}_lag_2"]
        current_features[f"{TARGET}_lag_2"] = current_features[f"{TARGET}_lag_1"]
        current_features[f"{TARGET}_lag_1"] = pred

    return forecasts
