import streamlit as st
import pickle
import predictor
import pandas as pd

# Load dropdowns
with open("dropdown_lists.pkl", "rb") as f:
    dropdowns = pickle.load(f)

df = predictor.raw_data

st.title("🇮🇳 T20 Match Predictor")
st.sidebar.header("Enter Match Details")

venue = st.sidebar.selectbox("Venue", dropdowns["venue_list"])
opponent = st.sidebar.selectbox("Opponent Team", dropdowns["team_list"])
toss_winner = st.sidebar.selectbox("Toss Winner", dropdowns["toss_winner_list"])
toss_decision = st.sidebar.radio("Toss Decision", ["bat", "field"])
target_score = st.sidebar.slider("Target Score", 100, 250, 160)

# Show historical stats based on opponent + target score
filtered = df[(df["team2"] == opponent) & (df["target_score"] == target_score)]
total = len(filtered)
india_wins = (filtered["winner"] == "India").sum()
opp_wins = (filtered["winner"] == opponent).sum()

st.markdown(f"### 📊 India vs **{opponent}**}")
if total > 0:
    st.markdown(f"- 🇮🇳 India Win %: **{(india_wins / total * 100):.1f}%**")
    st.markdown(f"- 🏴 {opponent} Win %: **{(opp_wins / total * 100):.1f}%**")
else:
    st.warning("No past data for this target score and opponent.")

st.markdown("---")

# Prediction
if st.button("Predict Winner"):
    result = predictor.predict_match_winner(venue, opponent, toss_winner, toss_decision, target_score)
    st.success(f"🏆 Predicted Winner: **{result}**")
