import os
import duckdb
import pandas as pd
import streamlit as st
import altair as alt
from openai import OpenAI
import time
# Hide the sidebar (including the collapse/expand arrow)
hide_sidebar_style = """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# ----------------- Top Navigation -----------------
# ----------------- Top Navigation -----------------
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


# ----------------- App Config -----------------
st.set_page_config(page_title="Fantasy Football AI", layout="wide")
# if st.sidebar.button("Open Getting Started"):
#     try:
#         st.switch_page("pages/getting_started.py")
#     except Exception:
#         # st.write(Exception)
#         st.sidebar.warning("Use the sidebar pages menu to open Getting Started.")



# ----------------- Secrets / Keys -----------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- DB Connection -----------------
@st.cache_resource
def get_con():
    return duckdb.connect("fantasy.duckdb", read_only=True)

con = get_con()

# ----------------- View Validation -----------------
def view_exists(view_name: str) -> bool:
    try:
        q = f"SELECT * FROM information_schema.tables WHERE table_name = '{view_name}'"
        return not con.execute(q).df().empty
    except Exception:
        return False
FRIENDLY_COLS = {
    "player": "Player",
    "position": "Position",
    "year": "Season",
    "adp": "Average Draft Position",
    "AVG_ADP": "Average Draft Position",
    "ppg_diff": "PPR points per game over Expectation",
    "ppg_diff": "Points per game over Expectation",
    "average_probability_over": "Probability of Overperforming",
    "average_probability_under": "Probability of Underperforming",
    "average_probability_neutral": "Probability of Neutral Performance",
}

def display_renamed(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view with user-friendly column labels, for display only."""
    # Only rename columns that exist to avoid KeyErrors
    cols_to_use = {k: v for k, v in FRIENDLY_COLS.items() if k in df.columns}
    return df.rename(columns=cols_to_use)


have_preds = view_exists("v_predictions")
have_hist = view_exists("v_history")
if not (have_preds and have_hist):
    st.warning("Expected views v_predictions and v_history not found. Run your schema_setup to create them.")

# ----------------- Prompt Template -----------------
SCHEMA_GUIDE = """
You translate natural-language questions into safe SQL for DuckDB.

You may ONLY query these views and columns:

-- v_predictions --
  player, position, year, AVG_ADP, average_probability_over, average_probability_under, average_probability_neutral
-- v_history --
  player, position, year, AVG_ADP, ppg_diff
-- v_adp --
  player, position, year, adp

Rules:
- Any questions outside the scope of these views should return nothing.
- Return ONLY a SQL query, no backticks, no prose.
- Never write DDL/DML. SELECT only.
- Always include an ORDER BY and a LIMIT when listing items (default LIMIT 25 if user doesn’t specify).
- If year = 2025, ALWAYS query v_predictions (never v_history), even if the user omits the word “predicted”.
- If year <= 2024, use v_history.
- Years: column is 'year'. Positions: 'QB','RB','WR','TE' in 'position'.
- If user asks “top 10 overperformers from 2024”, use v_history where year=2024 ORDER BY ppg_diff DESC LIMIT 10.
- If user asks “top 10 overperformers from 2025”, use v_predictions ORDER BY average_probability_over DESC LIMIT 10.
- If user asks "top 10 underperformers from 2025", use v_predictions ORDER BY average_probability_under DESC LIMIT 10.
- If user asks "performance of justin herbert in 2024", use v_history where exact match on player AND year=2024.
- For ADP-related queries, use v_adp.
- If user asks for "ADP of Josh Allen from 2021 - 2025", use v_adp where exact match on player and year BETWEEN 2021 AND 2025 ORDER BY year. 
  If exact match does not exist, do a fuzzy match using LIKE operator with wildcards.
"""

def llm_sql(user_question: str) -> str:
    """Ask the model for SQL only."""
    prompt = f"{SCHEMA_GUIDE}\n\nUser: {user_question}\nSQL:"
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = resp.choices[0].message.content.strip()
    banned = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "COPY", "EXPORT"]
    if any(b in sql.upper() for b in banned):
        raise ValueError("Unsafe SQL generated.")
    return sql

def build_chart(df: pd.DataFrame):
    alt.data_transformers.disable_max_rows()
    lower = {c.lower(): c for c in df.columns}

    # ---------- HISTORICAL: scatter ppg_diff vs ADP ----------
    if "ppg_diff" in lower and ("avg_adp" in lower or "adp" in lower):
        adp_col = lower.get("avg_adp", lower.get("adp"))
        d = df.copy()
        d[adp_col] = pd.to_numeric(d[adp_col], errors="coerce")
        d["ppg_diff"] = pd.to_numeric(d["ppg_diff"], errors="coerce")
        d = d.dropna(subset=[adp_col, "ppg_diff"])

        scatter = (
            alt.Chart(d)
            .mark_circle(size=70, opacity=0.9)
            .encode(
                x=alt.X(f"{adp_col}:Q", title="ADP (lower = earlier pick)",
                        scale=alt.Scale(reverse=True),
                        axis=alt.Axis(grid=False)),
                y=alt.Y("ppg_diff:Q", title="PPG vs Expectation",
                        axis=alt.Axis(grid=False)),
                color=alt.Color("player:N", legend=None, scale=alt.Scale(scheme="dark2")),
                tooltip=list(d.columns),
            )
            .properties(height=450)
        )
        zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6,4]).encode(y="y:Q")
        return (scatter + zero_rule).interactive(), "ppg_diff_vs_adp"

    # ---------- ADP OVER YEARS: line + points ----------
    if ("year" in lower) and ("avg_adp" in lower or "adp" in lower) \
       and not any(c.lower().startswith("average_probability") for c in df.columns):
        year_col = lower["year"]
        adp_col = lower.get("avg_adp", lower.get("adp"))

        d = df.copy()
        # coerce types
        d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
        d[adp_col]  = pd.to_numeric(d[adp_col],  errors="coerce")
        d = d.dropna(subset=[year_col, adp_col])
        d = d.sort_values(year_col)

        # If multiple players slip in, color by player; otherwise single line
        color_encoding = alt.value("#1f6f4a")  # dark green default
        if "player" in d.columns and d["player"].nunique() > 1:
            color_encoding = alt.Color("player:N", legend=alt.Legend(title="Player"),
                                       scale=alt.Scale(scheme="dark2"))

        line = (
            alt.Chart(d)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X(f"{year_col}:O", title="Season", axis=alt.Axis(grid=False)),
                y=alt.Y(f"{adp_col}:Q",
                        title="ADP (lower = earlier pick)",
                        scale=alt.Scale(reverse=False),
                        axis=alt.Axis(grid=False)),
                color=color_encoding,
                tooltip=list(d.columns),
            )
            .properties(height=420)
        )
        return line.interactive(), "adp_over_years"

    # ---------- PREDICTIONS: bar (fallback) ----------
    prob_cols = [c for c in df.columns if c.lower().startswith("average_probability")]
    if "player" in df.columns and prob_cols:
        ycol = prob_cols[0]
        d = df.copy()
        d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
        d = d.dropna(subset=[ycol]).sort_values(ycol, ascending=False).head(50)
        chart = (
            alt.Chart(d)
            .mark_bar()
            .encode(
                x=alt.X("player:N", sort="-y", title="Player"),
                y=alt.Y(f"{ycol}:Q", title=ycol.replace("_", " ").title(),
                        axis=alt.Axis(grid=False)),
                tooltip=list(d.columns),
            )
            .properties(height=420)
        )
        return chart, ycol

    return None, None


def llm_summary(df: pd.DataFrame, user_question: str) -> str:
    """
    Summarize the query results in natural language using the AI agent.
    Keeps payload small and focuses on relevant columns.
    """
    try:
        # Keep result small for the model (first 30 rows, drop super-wide tables)
        cols_priority = [c for c in df.columns if c.lower() in (
            "player","position","year","avg_adp","adp",
            "ppg_diff","average_probability_over","average_probability_under","average_probability_neutral"
        )]
        cols_use = cols_priority or list(df.columns[:10])  # fallback: first 10 columns
        df_small = df[cols_use].head(30)

        data_json = df_small.to_json(orient="records")

        prompt = f"""
You are an assistant that summarizes fantasy football query results into a concise,
user-friendly paragraph or short bullet list.

User question:
{user_question}

Data (JSON records):
{data_json}

Guidelines:
- If results include probabilities, mention which probability column was returned and the top 3–5 players.
- If results include ppg_diff, explain over vs under expectation and highlight top 3–5 values.
- Mention year(s) or position(s) if clearly present in the data.
- Keep it brief and helpful, no tables, no code.
"""

        resp = client.chat.completions.create(
            model="gpt-4.1-nano",   # use nano if you prefer lower cost
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"_Summary unavailable: {e}_"

# ----------------- Session State -----------------
if "last_df" not in st.session_state:
    st.session_state.last_df = None

# ----------------- UI -----------------
st.title("🏈 Fantasy Football AI Performance Predictor")

colL, colR = st.columns([2.5, 2.5])
with colL:
    input_col, button_col = st.columns([6, 0.75])
    with input_col:
        question = st.text_input(
            "Ask about fantasy football performance predictions and historical performances (2016-2024)",
            placeholder="show me the top 10 quarterbacks who overperformed in 2018",
        )
    with button_col:
        st.write("")
        st.write("")
        run = st.button("Run", use_container_width=True)

with colR:
    st.markdown("**Examples**")
    st.code("top 10 predicted underperforming wide receivers with an adp lower than 50 in 2025")
    st.code("worst 15 underperformers in 2019")
    st.code("Average Draft Position (ADP) of Joe Burrow from 2022 - 2024")
    st.code("how did Derrick Henry perform in 2024")

# ----------------- Query Execution -----------------
if run and question:
    try:
        # start_time = time.time()  # ⏱️ Start timer

        sql = llm_sql(question)
        df = con.execute(sql).df()
        # end_time = time.time()  # ⏱️ End timer
        # st.success(f"Query executed in {end_time - start_time:.2f} seconds.")
        st.session_state.last_df = df
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.last_df = None

# ----------------- Results Display -----------------
df = st.session_state.last_df
if df is None:
    pass
else:
    st.subheader("Results")
    st.subheader(question)
    with st.spinner("Generating summary…"):
        summary_text = llm_summary(df, question)
    # st.markdown("### Summary")
    st.markdown(summary_text)

    view_mode = st.radio(
        "View",
        options=["Table", "Chart"],
        index=0,
        horizontal=True,
        key="view_mode_main"
    )

    if view_mode == "Table":
        # show friendly labels **only in the table**
        st.dataframe(display_renamed(df), use_container_width=True)
    else:
        # keep the original df (raw column names) for charts
        chart, used_y = build_chart(df)
        if chart is None:
            st.warning("No chartable column found. Showing table instead.")
            st.dataframe(display_renamed(df), use_container_width=True)
        else:
            st.altair_chart(chart, use_container_width=True)

    # # ----------------- Explanations -----------------
    # cols = set(c.lower() for c in df.columns)
    # if "ppg_diff" in cols:
    #     st.markdown(
    #         """
    #         **Explanation**  
    #         • `ppg_diff` = Actual fantasy points per game − Expected points per game (from ADP).  
    #         • Positive → **overperformed** · Negative → **underperformed**.  
    #         • PPR scoring; rookies excluded due to limited prior signal.
    #         """
    #     )
    # elif any(c.startswith("average_probability") for c in df.columns):
    #     st.markdown(
    #         """
    #         **Explanation**  
    #         • Model-estimated probabilities:  
    #           – **Overperform** (`average_probability_over`)  
    #           – **Underperform** (`average_probability_under`)  
    #           – **Neutral** (`average_probability_neutral`)  
    #         • PPR scoring; rookies excluded due to limited prior signal.
    #         """
    #     )