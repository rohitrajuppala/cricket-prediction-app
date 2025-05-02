import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# In-memory dataset where Team1 is always India
data = [
    {
        "venue": "MCG",
        "team1": "India",
        "team2": "Australia",
        "toss_winner": "India",
        "toss_decision": "bat",
        "target_score": 180,
        "winner": "India"
    },
    {
        "venue": "SCG",
        "team1": "India",
        "team2": "England",
        "toss_winner": "England",
        "toss_decision": "field",
        "target_score": 150,
        "winner": "England"
    },
    {
        "venue": "Eden Gardens",
        "team1": "India",
        "team2": "Pakistan",
        "toss_winner": "Pakistan",
        "toss_decision": "bat",
        "target_score": 175,
        "winner": "India"
    },
    # Add more rows or load from your preprocessed India-only data
]

df = pd.DataFrame(data)
X = df.drop(columns=["winner"])
y = df["winner"]

categorical_cols = ["venue", "team1", "team2", "toss_winner", "toss_decision"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=25, max_depth=10, random_state=42))
])

model.fit(X, y)

def predict_match_winner(venue, team2, toss_winner, toss_decision, target_score):
    input_df = pd.DataFrame([{
        "venue": venue,
        "team1": "India",
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "target_score": target_score
    }])
    return model.predict(input_df)[0]

# Make training data accessible to the UI
raw_data = df
