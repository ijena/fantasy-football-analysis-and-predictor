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
Hey! I am Idhant Jena, a Class of 2025 undergraduate in Computer Science with a specialization in Intelligent Systems from the University of California, Irvine.
I am passionate about the intersection of AI and sports analytics, and this project is a testament to that passion. I am
currently looking for full-time opportunities in data science, software engineering, AI and Product Management. If you like what I have built and want to report any issues or feedback or just want to chat, 
please connect with me on [LinkedIn](https://www.linkedin.com/in/idhantjena/) or [Email](mailto:idhantjena7@gmail.com) 

---



---

### Sources


---
""")
