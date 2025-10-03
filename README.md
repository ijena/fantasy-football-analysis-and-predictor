**Fantasy Football AI — Over/Underperformance vs ADP (Agentic AI + DuckDB + Streamlit)
**
Predict which NFL players will overperform or underperform their Average Draft Position (ADP) — and explore historical performance — with an agentic AI interface that turns natural-language questions into safe SQL over DuckDB.
Built with Python, XGBoost/Random Forest, DuckDB, Streamlit, and OpenAI.

**Live App: https://fantasy-football-analysis-and-predictor.streamlit.app/
**
**✨ What this app does
**
Ask in plain English (agentic AI):
“Top 10 predicted overperforming WRs in 2025 with ADP < 50” → instant table, chart, and summary.

Predictions (2025): model-averaged probabilities of over / under / neutral vs ADP.

History (2016–2024): actual PPG vs expectation (derived from historical ADP models).

Visuals: sortable tables, probability bars, ADP vs PPG-diff scatter plots, ADP history per player line charts.

**🧠 Agentic AI (how the “agent” works)
**
A lightweight agent (OpenAI chat model) translates your question into read-only SQL against DuckDB views.

The agent is schema-constrained (prompted with allowed views/columns) and guard-railed (DDL/unsafe keywords blocked).

Results are summarized in natural language above the table/chart.

Models used in the app: gpt-4.1-nano (swap to your preferred model).

**📊 Data & Features
**
Data sources: FantasyPros (ADP), Pro-Football-Reference, NFLVerse (advanced stats).

Expected PPR (PPG): derived from historical ADP vs outcomes (leak-safe, LOESS/binned fallback).

Labels:

Overperform: actual PPG ≥ expected + threshold (or ≥ 80th percentile vs expected)

Underperform: actual PPG ≤ expected − threshold (or ≤ 20th percentile)

Neutral: everything else

Positions: QB, RB, WR, TE (kickers/IDP excluded). Rookies excluded (limited prior signal).

Scoring: PPR only.

**🏗️ Architecture
**CSVs → Pandas preprocessing → Feature engineering
         ↓
  Joblib models (RF/XGB) → 2025 predictions
         ↓
  DuckDB (tables + views): v_predictions, v_history, v_adp
         ↓
  Streamlit UI + Agentic AI (LLM → SQL → results → charts + summary)

**🧪 Reproducible Setup (Local)
**1) Clone & install
git clone https://github.com/<you>/fantasy-football-analysis-and-predictor.git
cd fantasy-football-analysis-and-predictor
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

2) Environment variables

Create .env or set Streamlit secrets with your OpenAI key:

OPENAI_API_KEY=sk-...

3) Load DuckDB tables

Point the loader to your CSV locations (predictions + history) and run:

python src/load_duckdb.py

This creates fantasy.duckdb, tables predictions, history, and views:

v_predictions(player, position, year, AVG_ADP, average_probability_over, average_probability_under, average_probability_neutral)

v_history(player, position, year, AVG_ADP, ppg_diff)

v_adp(player, position, year, adp)

4) Run the app
streamlit run src/streamlit_app.py

**🖥️ Deploy (Streamlit Community Cloud)
**
Push repo to GitHub (include requirements.txt and fantasy.duckdb OR the CSVs + loader).

In Streamlit → “New app” → select this repo and src/streamlit_app.py.

Add Secrets: OPENAI_API_KEY.

If you ship CSVs, add a postDeploy step or call load_duckdb.py on first run.

**🔍 Example questions
**
Predictions (2025)

“top 10 predicted overperformers for 2025”

“top 10 predicted underperforming WRs with adp lower than 50 in 2025”

History (2016–2024)

“show me the top 10 quarterbacks who overperformed in 2018”

“how did Derrick Henry perform in 2024”

“ADP of Joe Burrow from 2022–2024”

**📁 Repo structure
**src/
  ├─ streamlit_app.py        # main app (agent + UI)
  ├─ schema_setup.py         # creates views (optional if load script builds them)
  ├─ load_duckdb.py          # reads CSVs -> DuckDB tables + views
  ├─ prediction_2025.py      # batch predictions (optional)
  └─ pages/
      ├─ getting_started.py
      ├─ how_it_works.py
      └─ about.py
data/
  ├─ predictions_2025/final_model_2025.csv
  └─ historic_data/merged_historic_data.csv
models/
  ├─ qb_model_random_forest_classification.pkl
  └─ ... (other joblib models)
fantasy.duckdb               # built DB (or build on deploy)
requirements.txt
README.md

**🧩 How it works
**
Expected PPR per game is estimated from historical ADP vs outcomes with leak-safe fitting (prior years only).

Classification models (RF/XGB) predict over/under/neutral vs expectation.

Multiple models are ensembled by averaging probabilities per player.

Agentic AI converts questions → SQL for DuckDB → app renders tables, charts, and a natural-language summary.

**📈 Visualizations
**
Probability bars: over_performance, under_performance, neutral_performance.

Historical scatter: ppg_diff vs ADP (lower ADP = earlier pick).

ADP trajectory: player’s ADP by year line chart.

**⚠️ Limitations
**
PPR only; no kickers/defensive players; rookies excluded.

**🙋‍♂️ About the author
**
Built by Idhant Jena (UCI, CS — Intelligent Systems).
I’m actively looking for full-time roles in Software Engineering, AI/Data Science, and Product Management.
Connect: LinkedIn - https://www.linkedin.com/in/idhant-jena/
