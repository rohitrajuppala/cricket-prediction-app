import streamlit as st
import pickle
import predictor  # assumes you renamed predictor_team1_india_only.py to predictor.py

# Load dropdown lists
with open("dropdown_lists.pkl", "rb") as f:
    dropdowns = pickle.load(f)

st.title("🇮🇳 T20 Match Winner Predictor (India as Team 1)")

st.sidebar.header("Match Details")

venue = st.sidebar.selectbox("Select Venue", dropdowns["venue_list"])
opponent = st.sidebar.selectbox("Select Opponent Team", dropdowns["team_list"])
toss_winner = st.sidebar.selectbox("Select Toss Winner", dropdowns["toss_winner_list"])
toss_decision = st.sidebar.radio("Toss Decision", ["bat", "field"])

if st.button("Predict Winner"):
    result = predictor.predict_match_winner(venue, opponent, toss_winner, toss_decision)
    st.success(f"🏆 Predicted Winner: **{result}**")