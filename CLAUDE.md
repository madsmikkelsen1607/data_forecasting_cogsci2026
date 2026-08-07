# Project: Data Science, Prediction & Forecasting — Exam Project

## Context
Take-home exam for the course "Data Science, Prediction and Forecasting"
(Cognitive Science Master's). Solo paper, 10-12 standard pages. Topic and
method must be approved by the teacher.

Course grading emphasizes: describing/contrasting methods for time series,
identifying and preparing relevant data, visualizing time series, presenting
results to non-specialists, critically evaluating method appropriateness,
and discussing ethical/social/policy implications.

## Dataset
Rossmann Store Sales (Kaggle): daily sales for 1,115 drug stores in Germany.
Files: train.csv (daily sales per store), store.csv (store metadata: StoreType,
Assortment, Promo2, CompetitionDistance, etc).

## Scope decision
Full dataset is too large for a 10-12 page solo paper. Scoping via a
**stratified subset of stores**, not random or single-store — selected to
cover meaningful variation in StoreType, Assortment, Promo2 (continuous
promotions), and CompetitionDistance, so results can be discussed in terms
of how forecast performance differs across store characteristics. This
scoping rationale should be stated explicitly early in the paper.

## Planned workflow
1. Inspect raw data: date range, missingness, closed-store days, meaning of
   each column.
2. Cross-tabulate store.csv on StoreType / Assortment / Promo2 /
   CompetitionDistance; select handful of stores covering key combinations;
   write justification for each pick.
3. Build clean daily time series per selected store (decide + document
   handling of closed days).
4. EDA: trend, weekly seasonality, holiday/promo effects, stationarity checks.
5. Compare 2-3 forecasting methods (e.g. seasonal-naive baseline, SARIMA,
   Prophet or a lag-feature ML model) using a time-respecting train/val/test
   split (no shuffling) and a shared metric (RMSE or MAE).
6. Discuss ethical/social/policy angle: e.g. how forecast-driven staffing or
   inventory decisions affect store-level workers, fairness of demand
   allocation across store types/locations.

## Conventions
- Language/tooling: not yet decided in this file — update once chosen
  (e.g. Python version, key libraries).
- Keep commits scoped to one step of the workflow above where practical.

## Status
Scope and dataset finalized. Not yet started on implementation.
