import joblib
import pandas as pd

# Load trained model
model = joblib.load("t20_model.pkl")

# Encoding maps or schema should match training-time format
# This is a basic example assuming all categorical features were one-hot encoded
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
    df_encoded = pd.get_dummies(df)  # Simplified; ideally reuse same feature columns as training

    # Align columns with training model
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Add missing columns as 0
    df_encoded = df_encoded[model_columns]

    prediction = model.predict(df_encoded)[0]
    return prediction