import streamlit as st
import pickle
import predictor
import pandas as pd
import matplotlib.pyplot as plt

with open("dropdown_lists.pkl", "rb") as f:
    dropdowns = pickle.load(f)

st.title("🇮🇳 India T20 Match Win Predictor")
st.sidebar.header("Match Setup")

venue = st.sidebar.selectbox("Select Venue", dropdowns["venue_list"])
opponent = st.sidebar.selectbox("Select Opponent", [t for t in dropdowns["team_list"] if t != "India"])
toss_outcome = st.sidebar.selectbox("Toss Winner", ["India", opponent])
toss_decision = st.sidebar.radio("Toss Decision", ["bat", "field"])
target_score = st.sidebar.slider("Target Score", 50, 500, 200)

# Prediction
probs = predictor.predict_match_winner(venue, opponent, toss_outcome, toss_decision, target_score)
winner_classes = predictor.model.named_steps["classifier"].classes_

win_dict = dict(zip(winner_classes, probs))
india_pct = round(win_dict.get("India", 0) * 100, 1)
opp_pct = round(win_dict.get(opponent, 0) * 100, 1)

st.markdown("## 📊 Predicted Winning Chances")
st.write(f"- 🇮🇳 India Win %: **{india_pct}%**")
st.write(f"- 🏴 {opponent} Win %: **{opp_pct}%**")

# EDA
st.markdown("## 📈 India Win % by Target Score Range")
eda_df = predictor.win_stats
team_data = eda_df[eda_df["team2"] == opponent]

if not team_data.empty:
    fig, ax = plt.subplots()
    team_data.plot(x="target_bin", y="India", kind="bar", ax=ax, legend=False)
    plt.ylabel("India Win %")
    plt.title(f"India Win % vs {opponent} by Target Score Range")
    st.pyplot(fig)
else:
    st.info("No historical data for this team and target score combination.")
