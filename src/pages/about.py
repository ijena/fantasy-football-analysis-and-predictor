# pages/about.py
import streamlit as st

# Page config + hide sidebar
st.set_page_config(page_title="About — Fantasy Football AI", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# --- Top Navigation ---
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

# --- Page Content ---
st.title("ℹ️ About")

st.markdown("""
### Project Overview
This app is an **agentic AI assistant** built on top of:
- **DuckDB** for data storage  
- **Machine learning models** predicting fantasy football over/underperformance  
- **Streamlit** for interactive visualizations  
- **OpenAI models** for natural-language → SQL translation and summarization  

It allows anyone to ask free-form questions like:  
- *“Top 10 predicted overperformers for 2025”*  
- *“How did Derrick Henry perform in 2024?”*  
- *“Average Draft Position of Joe Burrow from 2021–2024”*  

and get results as **tables, charts, and natural-language summaries**.

---

### Creator
This project was developed by **[Your Name]**, combining my interests in:
- Sports analytics ⚽🏈  
- Machine learning 🤖  
- Interactive data visualization 📊  
- Building real-world agentic AI applications  

---

### Open Source & Sharing
- Code hosted on GitHub (link if you want).  
- Free to use and explore on Streamlit Community Cloud.  

---
""")
