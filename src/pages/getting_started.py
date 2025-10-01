import streamlit as st

# Hide the sidebar (including the collapse/expand arrow)
hide_sidebar_style = """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)
HOME_PAGE = "streamlit_app.py"              # your main file
GETTING_STARTED_PAGE = "pages/getting_started.py"  # rename to match your file

# Make sure the page exists

    # Build a horizontal nav bar
nav_col1, nav_col2,_ = st.columns([0.10, 0.20,0.80],gap='small')
with nav_col1:
    if st.button("🏠 Home"):
            st.switch_page(HOME_PAGE)
with nav_col2:
    if st.button("📘 Getting Started"):
            st.switch_page(GETTING_STARTED_PAGE)
st.set_page_config(page_title="Getting Started — Fantasy Football AI", layout="wide")

st.title("📘Getting Started")
st.markdown("""
Welcome! This web app answers fantasy football player performance questions while also giving you historical player performance data

---

## How to use this tool
- **Predictions (2025):**
  - Ask questions about predicted overperformers and underperformers for the 2025 season
  - You can filter by player name, position, average draft position (ADP) and number of results
  - Examples:
  - `top 10 predicted overperformers for 2025`
  - `who are the top 20 predicted underperforming wide receivers with an adp lower than 50 in 2025`
  - `how will Josh Allen perform in 2025`

- **Historical data (2016–2024):**
  - Ask questions about overperformers and underperformers from 2016 to 2024
  - You can also ask questions about specific players in specific years to see their performance and average draft position (ADP)
  - You can filter by player name, position, average draft position (ADP) and number of results
  - Examples:
  - `show me the top 10 quarterbacks who overperformed in 2018`
  - `worst 15 underperformers in 2019`
  - `Average Draft Position (ADP) of Joe Burrow from 2022 - 2024'
  - 'how did Derrick Henry perform in 2024'

## Constraints
- Predictions and history are for **Points Per Reception (PPR)** format.
- Rookies are excluded due to lack of historical NFL data.
- Kickers and defensive players are excluded.
- Historical data is only from 2016 to 2024.
- If you ask outside the supported views, the AI should return no results.

---

""")
