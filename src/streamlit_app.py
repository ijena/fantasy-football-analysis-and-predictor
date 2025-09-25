import os
import json
import duckdb
import pandas as pd
import streamlit as st
import altair as alt

# --- OpenAI client (official SDK v1) ---
from openai import OpenAI

# ----------------- App Config -----------------
st.set_page_config(page_title="Fantasy Football AI", layout="wide")

# ----------------- Secrets / Keys -----------------
# On Streamlit Cloud, set in: Settings → Secrets as:
# OPENAI_API_KEY = "sk-..."
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- DB Connection -----------------
@st.cache_resource
def get_con():
    # fantasy.duckdb should be in the repo root
    return duckdb.connect("fantasy.duckdb", read_only=True)

con = get_con()

# Optional: validate views exist
def view_exists(view_name: str) -> bool:
    try:
        q = f"SELECT * FROM information_schema.tables WHERE table_name = '{view_name}'"
        return not con.execute(q).df().empty
    except:
        return False

have_preds = view_exists("v_predictions")
have_hist  = view_exists("v_history")

if not (have_preds and have_hist):
    st.warning("Expected views v_predictions and v_history not found. "
               "Run your schema_setup to create them.")
    # You can still continue if you know the tables are named differently.

# ----------------- Prompt Template -----------------
SCHEMA_GUIDE = """
You translate natural-language questions into safe SQL for DuckDB.

You may ONLY query these views and columns:

-- v_predictions --
  player, position, year, AVG_ADP,average_probability_over,average_probability_under,average_probability_neutral
-- v_history --
  player, position, year, AVG_ADP, ppg_diff
--v_adp --
    player, position, year, adp

Rules:
- Return ONLY a SQL query, no backticks, no prose.
- Never write DDL/DML. SELECT only.
- Always include an ORDER BY and a LIMIT when listing items (default LIMIT 25 if user doesn’t specify).
- For "top overperformers/underperformers", use v_history (ppg_diff) or probabilities in v_predictions.
- Years: column is 'year'. Positions: 'QB','RB','WR','TE' in 'position'.
- If user asks “top 10 overperformers from 2024”, use v_history where year =2024,ORDER BY ppg_diff DESC LIMIT 10.
- If user asks “top 10 overperformers from 2025”, use v_predictions ORDER BY average_probability_over  and only show average_probability_over among the probability columns DESC LIMIT 10.
- If user asks "top 10 underperformers from 2025", use v_predictions ORDER BY average_probability_under and only show average_probability_under among the probability columns DESC LIMIT 10.
- For ADP-related queries, use v_adp.
- If user asks for "ADP of Josh Allen from 2021 - 3025", use v_adp where exact match on player and year BETWEEN 2021 AND 2025 ORDER BY year. If exact match does not exist, do a fuzzy match using LIKE operator with wildcards.
"""

def llm_sql(user_question: str) -> str:
    """Ask the model for SQL only."""
    prompt = f"{SCHEMA_GUIDE}\n\nUser: {user_question}\nSQL:"
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = resp.choices[0].message.content.strip()
    # quick guard: ban dangerous keywords
    banned = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "COPY", "EXPORT"]
    if any(b in sql.upper() for b in banned):
        raise ValueError("Unsafe SQL generated.")
    return sql

# ----------------- UI -----------------
st.title("🏈 Fantasy Football AI Performance Predictor and Historic Data")

colL, colR = st.columns([3, 2])
with colL:
    question = st.text_input(
        "Ask a question about fantasy football performance predictions and historical performances",
        placeholder="show me the top 10 quarterbacks who overperformed in 2018",
    )

with colR:
    st.markdown("**Examples**")
    st.code("top 10 predicted quarterback overperformers for 2025")
    st.code("worst 15 underperformers among WR in 2019")
    st.code("ADP of Joe Burrow from 2021 - 2025")
    
run = st.button("Run")

if run and question:
    try:
        sql = llm_sql(question)
        # st.subheader("Generated SQL")
        # st.code(sql, language="sql")

        df = con.execute(sql).df()
        if df.empty:
            st.info("No rows returned.")
        else:
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            # Simple heuristics to chart smartly
            cols = set(c.lower() for c in df.columns)
            if "player" in cols and "ppg_diff" in cols:
                chart = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("player:N", sort="-y"),
                        y=alt.Y("ppg_diff:Q"),
                        tooltip=list(df.columns),
                    )
                    .properties(height=400)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                # Try probability bar if present
                prob_cols = [c for c in df.columns if c.lower().startswith("average_")]
                if "player" in df.columns and prob_cols:
                    ycol = prob_cols[0]
                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("player:N", sort="-y"),
                            y=alt.Y(f"{ycol}:Q"),
                            tooltip=list(df.columns),
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

