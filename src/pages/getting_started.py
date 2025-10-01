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

st.title("🚀 Getting Started")
st.markdown("""
Welcome! This app lets you **ask natural-language questions** about fantasy football:
- **Predictions for 2025** (probabilities of *over/under/neutral* vs expectations)
- **Historical over/under-performance** (2016–2024)
- **ADP lookups** for specific players and ranges

---

## How it works (in plain English)
- You type a question.
- An **agentic AI** translates your question into **safe SQL** that runs against a local DuckDB file.
- Results are shown as a **table** or a **chart** (toggle at the top of the results).

Behind the scenes:
- **Predictions (2025)** come from your saved models. The app queries `v_predictions`.
- **History (≤ 2024)** comes from past seasons. The app queries `v_history`.
- **ADP** queries use `v_adp`.

---

## Examples you can paste
- **Predictions (2025):**
  - `top 10 predicted overperformers for 2025`
  - `top 10 predicted underperforming wide receivers with an adp lower than 50 in 2025`
  - `top 15 predicted overperforming QBs in 2025`

- **Historical (2016–2024):**
  - `show me the top 10 quarterbacks who overperformed in 2018`
  - `worst 15 underperformers in 2019`
  - `overperformers among WR in 2020 (top 20)`

- **ADP lookups:**
  - `Average Draft Position (ADP) of Joe Burrow from 2022 - 2024`
  - `ADP of Justin Jefferson in 2021`

> Tip: If a player name doesn’t match exactly, the agent will try fuzzy matching.

---

## What the columns mean
- **ppg_diff**: `Actual PPR points per game – Expected PPR (from ADP)`  
  - Positive ⇒ **overperformed**  
  - Negative ⇒ **underperformed**
- **average_probability_over / under / neutral**: model-estimated probabilities for 2025.

---

## Known constraints
- **Rookies** are excluded (insufficient prior signal for ADP-based expectations).
- Predictions and history are for **PPR** format.
- If you ask for 2025 history, expect **no rows** (use predictions instead).
- If you ask outside the supported views, the AI should return no results.

---

## Troubleshooting
- **No rows returned?**  
  - Check the year you asked for (history ≤ 2024, predictions = 2025).
  - Try a broader query (e.g., drop position filters).
- **No chart appears?**  
  - Some queries don’t return a numeric column the chart expects. Switch to “Table” view.

---

## Credits & Contact
- Models, data prep, and app by **you** 🙌  
- Engine: DuckDB + Streamlit + an agent that translates English → SQL  
- Questions or ideas? Open an issue on the repo or reach out on LinkedIn.
""")
