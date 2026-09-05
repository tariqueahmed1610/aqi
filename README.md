# AQI Predictor
**Live Dashboard:** https://aqi-predictor-tarique.streamlit.app/
Predicts Air Quality Index 1, 2, and 3 days ahead for 5 Pakistani cities
(Karachi, Lahore, Islamabad, Peshawar, Quetta), using a serverless
feature store / model registry / dashboard architecture.
## Architecture
- **Feature pipeline** (`feature_pipeline.py`) — pulls weather + pollutant
  data from Open-Meteo, engineers features, writes to a Hopsworks
  Feature Group. Runs hourly via GitHub Actions.
- **Training pipeline** (`training_pipeline.py`) — reads from the
  Hopsworks Feature Store, trains/compares Ridge, Random Forest, and
  XGBoost for each forecast horizon (24h/48h/72h), registers the best
  model per horizon. Runs daily via GitHub Actions.
- **Dashboard** (`streamlit_app.py`) — loads the latest features +
  registered models, shows a 3-day forecast per city. Deployed live on
  Streamlit Community Cloud.
## Setup
1. Clone this repo.
2. `pip install -r requirements.txt`
3. Set the `HOPSWORKS_API_KEY` environment variable (never commit it).
4. Run `python feature_pipeline.py` once with `BACKFILL=True` to build
   history, then `python training_pipeline.py`.
## Automation
See `.github/workflows/` for the scheduled jobs that keep the feature
store and models up to date without any manual runs.
