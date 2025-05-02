import streamlit as st
import pickle
import predictor

# Load dropdown lists
with open("dropdown_lists.pkl", "rb") as f:
    dropdowns = pickle.load(f)

st.title("🏏 T20 Match Winner Predictor")

st.sidebar.header("Match Details")

venue = st.sidebar.selectbox("Select Venue", dropdowns["venue_list"])
team1 = st.sidebar.selectbox("Select Team 1", dropdowns["team_list"])
team2 = st.sidebar.selectbox("Select Opponent Team", dropdowns["team_list"])
toss_winner = st.sidebar.selectbox("Select Toss Winner", dropdowns["toss_winner_list"])
bat_first = st.sidebar.radio("Toss Decision", ["bat", "field"])

if st.button("Predict Winner"):
    result = predictor.predict_match_winner(venue, team1, team2, toss_winner, bat_first)
    st.success(f"🏆 Predicted Winner: **{result}**")