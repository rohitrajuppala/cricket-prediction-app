import pandas as pd
import joblib

# Load the compressed model trained in a Python 3.12-compatible environment
model = joblib.load("t20_model.pkl")

def predict_match_winner(venue, team1, team2, toss_winner, toss_decision):
    input_dict = {
        "venue": venue,
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision
    }

    df = pd.DataFrame([input_dict])

    # Ensure all encoding, feature engineering matches training-time logic
    df_encoded = pd.get_dummies(df)  # Simplified; assumes model trained with one-hot encoding

    # Align with training feature columns
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_columns]

    prediction = model.predict(df_encoded)[0]
    return prediction
