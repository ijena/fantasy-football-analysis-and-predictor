import os
import duckdb
import pandas as pd
import streamlit as st
import altair as alt
from openai import OpenAI

# ----------------- App Config -----------------
st.set_page_config(page_title="Fantasy Football AI", layout="wide")
st.sidebar.markdown("### 📘 Help")
st.sidebar.page_link("pages/getting_started.py", label="Getting Started")


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

# ----------------- Chart Builder -----------------
def build_chart(df: pd.DataFrame):
    alt.data_transformers.disable_max_rows()
    lower = {c.lower(): c for c in df.columns}

    # Historical performance
    if "ppg_diff" in lower and "player" in lower:
        ycol = lower["ppg_diff"]
    else:
        # Prediction probabilities
        prob_cols = [c for c in df.columns if c.lower().startswith("average_probability")]
        ycol = prob_cols[0] if ("player" in df.columns and prob_cols) else None

    if ycol is None or "player" not in df.columns:
        return None, None

    d = df.copy()
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[ycol])
    if d.empty:
        return None, ycol

    if "year" in d.columns and pd.api.types.is_float_dtype(d["year"]):
        d["year"] = d["year"].astype("Int64")

    d = d.sort_values(ycol, ascending=False).head(50)

    title_map = {
        "ppg_diff": "PPG vs Expectation",
        "average_probability_over": "Probability: Overperform",
        "average_probability_under": "Probability: Underperform",
        "average_probability_neutral": "Probability: Neutral",
    }
    y_title = title_map.get(ycol, ycol.replace("_", " ").title())

    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("player:N", sort="-y", title="Player"),
            y=alt.Y(f"{ycol}:Q", title=y_title),
            tooltip=list(d.columns),
        )
        .properties(height=420)
    )
    return chart, ycol

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
        sql = llm_sql(question)
        df = con.execute(sql).df()
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

    view_mode = st.radio(
        "View",
        options=["Table", "Chart"],
        index=0,
        horizontal=True,
        key="view_mode_main"
    )

    if view_mode == "Table":
        st.dataframe(df, use_container_width=True)
    else:
        chart, used_y = build_chart(df)
        if chart is None:
            st.warning("No chartable column found. Showing table instead.")
            st.dataframe(df, use_container_width=True)
        else:
            st.altair_chart(chart, use_container_width=True)

    # ----------------- Explanations -----------------
    cols = set(c.lower() for c in df.columns)
    if "ppg_diff" in cols:
        st.markdown(
            """
            **Explanation**  
            • `ppg_diff` = Actual fantasy points per game − Expected points per game (from ADP).  
            • Positive → **overperformed** · Negative → **underperformed**.  
            • PPR scoring; rookies excluded due to limited prior signal.
            """
        )
    elif any(c.startswith("average_probability") for c in df.columns):
        st.markdown(
            """
            **Explanation**  
            • Model-estimated probabilities:  
              – **Overperform** (`average_probability_over`)  
              – **Underperform** (`average_probability_under`)  
              – **Neutral** (`average_probability_neutral`)  
            • PPR scoring; rookies excluded due to limited prior signal.
            """
        )
