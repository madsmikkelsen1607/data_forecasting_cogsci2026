# Rossmann Sales Forecasting

Code accompanying the exam paper for Data Science, Prediction and Forecasting. 
Compares three forecasting methods of increasing
complexity - seasonal-naive, SARIMA, and a lag-feature gradient boosting
model - on daily sales for a stratified subset of Rossmann drugstore stores,
to ask whether forecast accuracy depends on store characteristics.

## Data

Uses the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
dataset (Kaggle, 2015). `train.csv` and `store.csv` are required in `data/`
and are not included in this repository (see `.gitignore`) — download them
from Kaggle and place them there before running any notebook.

## Repository structure

```
src/                  Reusable logic, imported by the notebooks
  data.py             Load and clean train.csv / store.csv
  store_selection.py  Cross-tabulation and stratified selection of 6 stores
  timeseries.py       Per-store trading-day series (closed days dropped)
  models.py           SeasonalNaive / SARIMA / lag-feature GBRT forecasters
  evaluate.py         Time-respecting train/val/test split, RMSE/MAE, comparison
  plotting.py         Shared chart style and table-image rendering

notebooks/             One notebook per stage of the analysis, in order:
  01_inspect_data.ipynb        Raw data overview
  02_store_selection.ipynb     Stratified selection of 6 stores
  03_eda.ipynb                 Trend, seasonality, stationarity checks
  04_modeling.ipynb            Fit and evaluate all three methods per store
  05_results_summary.ipynb     Cross-store comparison and figures

outputs/
  tables/             Generated CSVs (store selection, model comparison, ...)
  figures/            Generated charts and table images used in the paper
```

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Run the notebooks in numeric order (01 through 05); each depends on files
written to `outputs/` by the ones before it.
