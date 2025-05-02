import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load cleaned data with real target scores
with open("india_vs_matches.json") as f:
    data = json.load(f)

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
    ("classifier", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))
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
    return model.predict_proba(input_df)[0]

# EDA function: win % by target bins
def win_percent_by_target(df):
    bins = [0, 100, 200, 300, 400, 500]
    df["target_bin"] = pd.cut(df["target_score"], bins=bins)
    agg = df.groupby(["team2", "target_bin"])["winner"].value_counts(normalize=True).unstack().fillna(0)
    return agg.reset_index()

win_stats = win_percent_by_target(df)
raw_data = df
