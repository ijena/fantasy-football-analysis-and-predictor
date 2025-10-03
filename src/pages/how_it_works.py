# pages/how_it_works.py
import streamlit as st

# Page config + hide sidebar on this page too
st.set_page_config(page_title="How it works — Fantasy Football AI", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}

        /* Make only non-heading markdown text 17px */
        .stMarkdown p, .stMarkdown li {
            font-size: 17px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Top nav (same as main for consistency)
HOME_PAGE = "streamlit_app.py"
GETTING_STARTED_PAGE = "pages/getting_started.py"
HOW_IT_WORKS_PAGE = "pages/how_it_works.py"
ABOUT_PAGE = "pages/about.py"

nav_col1, nav_col2, nav_col3, nav_col4, _ = st.columns([0.10, 0.15, 0.15, 0.10, 0.70], gap="small")
with nav_col1:
    if st.button("🏠 Home"):
        st.switch_page(HOME_PAGE)
with nav_col2:
    if st.button("📘 Getting Started"):
        st.switch_page(GETTING_STARTED_PAGE)
with nav_col3:
    if st.button("🧠 How it works"):
        st.switch_page(HOW_IT_WORKS_PAGE)
with nav_col4:
    if st.button("ℹ️ About"):
        st.switch_page(ABOUT_PAGE)

# --------- Content ---------
st.title("🧠 How it works")

st.markdown(""" 
### 1) Data preprocessing using Pandas in Python  
- The web app uses fantasy football data from [FantasyPros](https://www.fantasypros.com) (ADP) and [Pro-Football-Reference](https://www.pro-football-reference.com) (player stats).  
- Additionally, I gathered NFL player season statistics and advanced data from [NFLVerse](https://nflverse.nflverse.com/).  
- All fantasy irrelevant data was excluded from these datasets (e.g., kickers, defensive players, offensive linemen, etc.) 
- I calculated average ADP for each player across different platforms (e.g., Sleeper, Yahoo) to get a consensus ADP for each player for each year.
- The datasets were merged with cleaning and normalization of player names and handling missing values, to create a unified view of player ADP and performance from 2015 to 2024.  
- Since Average Draft Position (ADP) defines player expectations in fantasy football, I calculated expected fantasy points per game based on ADP and player position using historical data.  
- This was used to engineer "performance relative to expectations" which is the difference between actual and expected points per game.  

### 2) Model training  
- The merged dataset were split by fantasy relevant positions (QB, RB, WR, TE).  
- The data was split into training (2016-2020), validation (2021-2023) and test (2024) sets based on season year.  
- Classification models (Random Forest, XGBoost) were trained in Python on the historical training data to predict whether a player will overperform, underperform, or meet expectations based on their ADP and fantasy performance and statistical data from the previous season.  
- Overperformance is defined as exceeding expected points per game by 2 points or by being in the 80th percentile of performance relative to expectations.  
- Underperformance is defined as failing to meet expected points per game by 2 points or less or by being in the 20th percentile of performance relative to expectations.  
- All other performances are classified as meeting expectations.  
- These models were fine-tuned using the validation set and evaluated on the test set to ensure they generalize well to unseen data.  

### 3) Model predictions  
- The trained models were saved using joblib for each type of model for each fantasy relevant position.  
- These models were used to generate predictions for the 2025 season based on Average Draft Position (ADP) data for the 2025 season.  
- The predictions include probabilities for each class (overperform, underperform, meet expectations) for each player based on their ADP.  
- The average probability for each class was calculated across all models for each player to create a consensus prediction.  

### 4) Data Storage (DuckDB)  
- The model predictions were stored in a DuckDB database called v_predictions, which included player details like name, position and ADP along with average probability of overperforming, underperforming and meeting expectations.  
- Historical performance data from 2016 to 2024 was also stored in a DuckDB view called v_history, which included player details, ADP and actual performance relative to expectations.  
- Player ADP data from 2015 to 2025 was stored in a DuckDB view called v_adp to allow the querying of player ADP.  

### 5) Model deployment (Streamlit + OpenAI)  
- The web app was built using Streamlit to provide an interactive interface for users to ask questions about player performance.  
- The user can ask questions about performance predictions based on 2025 ADP or player positions or ask for historical performance data from 2016 to 2024.  
- These natural language questions are processed using Agentic AI (OpenAI's GPT-4.1-nano model) to understand user intent and generate SQL queries to fetch relevant data from the DuckDB database.  
- The AI Agent has specific schema rules to ensure it generates safe and accurate SQL queries based on the user's question.  
- The results are then displayed in a user-friendly format, including tables and charts, along with a natural language summary generated by the AI Agent.  
""")
