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

st.set_page_config(page_title="Getting Started — Fantasy Football AI", layout="wide")

# Title
st.markdown("<h1 style='font-size:34px;'>📘 Getting Started</h1>", unsafe_allow_html=True)

# Intro
st.markdown(
    """
    <p style='font-size:20px;'>
    Welcome! This web app answers fantasy football questions in terms of expected 
    overperformance and underperformance of NFL players relative to their average draft position (ADP) in fantasy football drafts.  
    It can also answer historical questions about fantasy football player performance from 2016 to 2024.
    </p>
    <hr>
    """, unsafe_allow_html=True
)

# How to use
st.markdown("<h2 style='font-size:26px;'>How to use this tool</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <p style='font-size:18px;'><b>Predictions (2025):</b></p>
    <ul style='font-size:17px;'>
      <li>Ask questions about predicted overperformers and underperformers for the 2025 season</li>
      <li>You can ask questions that are filtered by player name, position, average draft position (ADP) and number of results</li>
      <li>Examples:</li>
      <ul>
        <li><code>top 10 predicted overperformers for 2025</code></li>
        <li><code>who are the top 20 predicted underperforming wide receivers with an adp lower than 50 in 2025</code></li>
        <li><code>how will Josh Allen perform in 2025</code></li>
      </ul>
    </ul>

    <p style='font-size:18px;'><b>Historical data (2016–2024):</b></p>
    <ul style='font-size:17px;'>
      <li>Ask questions about overperformers and underperformers from 2016 to 2024</li>
      <li>You can ask questions about specific players in specific years to see their performance and ADP</li>
      <li>You can filter by player name, position, ADP and number of results</li>
      <li>Examples:</li>
      <ul>
        <li><code>show me the top 10 quarterbacks who overperformed in 2018</code></li>
        <li><code>worst 15 underperformers in 2019</code></li>
        <li><code>Average Draft Position (ADP) of Joe Burrow from 2022 - 2024</code></li>
        <li><code>how did Derrick Henry perform in 2024</code></li>
      </ul>
    </ul>
    """,
    unsafe_allow_html=True
)

# Constraints
st.markdown("<h2 style='font-size:26px;'>Constraints</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <ul style='font-size:17px;'>
      <li>Predictions and history are only for <b>Points Per Reception (PPR)</b> format.</li>
      <li>Rookies are excluded due to lack of historical NFL data.</li>
      <li>Kickers and defensive players are excluded.</li>
      <li>Historical data is only from 2016 to 2024.</li>
    </ul>
    <hr>
    """, unsafe_allow_html=True
)
