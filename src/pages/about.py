# pages/about.py
import streamlit as st

# --- Page config + hide sidebar ---
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

# Profile + About in two columns
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.image("images/profile.JPG", width=220, caption="Idhant Jena")  # Replace with your image path

with col2:
    st.markdown("""
    Hey! I am Idhant Jena, a Class of 2025 undergraduate in Computer Science (Intelligent Systems) from the University of California, Irvine.  

    I am passionate about the intersection of AI and sports analytics, and this project is a testament to that passion.  

    I am currently seeking full-time opportunities in:  
    - Data Science  
    - Software Engineering  
    - Artificial Intelligence  
    - Product Management  

    📩 If you like what I have built and want to report any issues, share feedback, or just chat:  
    - [LinkedIn](https://www.linkedin.com/in/idhantjena/)  
    - [Email](mailto:idhantjena7@gmail.com)  
    """)

st.markdown("---")

# Full-width Sources section
st.subheader("### Sources")
st.markdown("""
- Fantasy football ADP & performance data sourced from [FantasyPros](https://www.fantasypros.com) and public datasets.  
- Predictions generated using fine-tuned ML models and OpenAI APIs.  
- Visualization powered by [Streamlit](https://streamlit.io) and [Altair](https://altair-viz.github.io).  
""")
