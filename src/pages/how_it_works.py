# pages/how_it_works.py

import streamlit as st

# Page config + hide sidebar on this page too
st.set_page_config(page_title="How it works — Fantasy Football AI", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Top nav (same as main for consistency)
HOME_PAGE = "streamlit_app.py"
GETTING_STARTED_PAGE = "pages/getting_started.py"
HOW_IT_WORKS_PAGE = "pages/how_it_works.py"

nav_col1, nav_col2, nav_col3, _ = st.columns([0.10, 0.15, 0.15, 0.80], gap="small")
with nav_col1:
    if st.button("🏠 Home"):
        st.switch_page(HOME_PAGE)
with nav_col2:
    if st.button("📘 Getting Started"):
        st.switch_page(GETTING_STARTED_PAGE)
with nav_col3:
    if st.button("🧠 How it works"):
        st.switch_page(HOW_IT_WORKS_PAGE)

# --------- Content ---------
st.title("🧠 How it works")

st.markdown("""
This app combines **agentic AI** with your **ML models** and a **DuckDB** warehouse to answer natural-language
questions about fantasy football performance.

### 1) Data & storage (DuckDB)
- You load two main datasets:
  - **Predictions (2025)** with model probabilities for Over / Under / Neutral vs expectation.
  - **History (2016–2024)** with actual performance vs expectation.
- We create read-only views:
  - `v_predictions(player, position, year, AVG_ADP, average_probability_over, average_probability_under, average_probability_neutral)`
  - `v_history(player, position, year, AVG_ADP, ppg_diff)`

### 2) Expected points & labels
- **Expectation** is derived from **ADP-based curves** trained on past seasons (leak-safe).
- **Overperformance** (historical) uses `ppg_diff = actual PPR per game − expected PPR per game`.
- **Predictions** (2025) are class probabilities from your saved models.

### 3) Agentic AI → SQL
- A small **prompting policy** tells the model how to translate your question into **safe SQL**:
  - 2025 → query `v_predictions` and sort by the right probability.
  - ≤ 2024 → query `v_history` and sort by `ppg_diff`.
  - ADP questions → query `v_adp` (if present).
- The app **disallows DDL/DML** and screens for unsafe keywords before running.

### 4) Results → Table, Chart, Summary
- **Table**: we show user-friendly column names (e.g., “Probability of Overperforming”).
- **Chart**: Altair bar charts for either `ppg_diff` (history) or the selected probability (predictions).
- **Natural-language summary**: A short model-generated explanation for context.

### 5) Why agentic?
- The “agent” takes your intent, **plans** the appropriate query, **executes** it, and **explains** the results.
- This is more than keyword search; it’s task-oriented orchestration across **NL → SQL → Viz → Summary**.

### 6) Limits & scope
- PPR scoring only; rookies excluded due to limited prior signal.
- 2016–2024 for history; 2025 for predictions.
- Names must exist in the datasets (we do simple fuzzy fallback).

---
**Tip:** Try asking:
- “Top 10 predicted underperforming WR with ADP < 50 in 2025”
- “Show me the biggest QB overperformers in 2018”
""")
