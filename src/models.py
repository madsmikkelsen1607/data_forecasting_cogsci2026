"""Three comparable forecasters sharing one interface: `fit(train)` then
`predict(horizon, future)`, where `train`/`future` are frames produced by
`timeseries.build_store_series` (trading-day indexed, with Date, Sales,
DayOfWeek, Promo, StateHoliday, SchoolHoliday columns).

1. SeasonalNaiveForecaster - repeats the last observed seasonal cycle.
2. SarimaForecaster - statsmodels SARIMAX, small AIC-based grid search over
   (p,d,q)(P,D,Q,s) instead of adding a pmdarima dependency.
3. LagFeatureMLForecaster - GradientBoosting/RandomForest on lag/rolling
   features of Sales plus the known-in-advance exogenous covariates
   (DayOfWeek, Promo, StateHoliday, SchoolHoliday), forecasting
   recursively step by step.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SeasonalNaiveForecaster:
    def __init__(self, seasonal_period: int):
        self.seasonal_period = seasonal_period
        self._last_season = None

    def fit(self, train: pd.DataFrame) -> "SeasonalNaiveForecaster":
        self._last_season = train["Sales"].to_numpy(dtype=float)[
            -self.seasonal_period :
        ]
        return self

    def predict(self, horizon: int, future: pd.DataFrame | None = None) -> np.ndarray:
        reps = int(np.ceil(horizon / self.seasonal_period))
        return np.tile(self._last_season, reps)[:horizon]


@dataclass
class SarimaForecaster:
    seasonal_period: int
    order_grid: list = field(
        default_factory=lambda: [
            (0, 1, 1),
            (1, 1, 0),
            (1, 1, 1),
            (2, 1, 1),
            (0, 1, 2),
        ]
    )
    seasonal_order_grid: list = field(
        default_factory=lambda: [(0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 0, 1)]
    )

    def __post_init__(self):
        self._fitted = None
        self.best_order = None
        self.best_seasonal_order = None
        self.best_aic = None

    def fit(self, train: pd.DataFrame) -> "SarimaForecaster":
        y = train["Sales"].astype(float)
        best_aic = np.inf
        best_res = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in self.order_grid:
                for sorder in self.seasonal_order_grid:
                    seasonal_order = (*sorder, self.seasonal_period)
                    try:
                        res = SARIMAX(
                            y,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    except Exception:
                        continue
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_res = res
                        self.best_order = order
                        self.best_seasonal_order = seasonal_order
        if best_res is None:
            raise RuntimeError("SARIMA grid search failed to converge on any order")
        self._fitted = best_res
        self.best_aic = best_aic
        return self

    def predict(self, horizon: int, future: pd.DataFrame | None = None) -> np.ndarray:
        return self._fitted.get_forecast(steps=horizon).predicted_mean.to_numpy()


class LagFeatureMLForecaster:
    def __init__(
        self,
        seasonal_period: int,
        model: str = "gbrt",
        n_lags: int | None = None,
        random_state: int = 42,
    ):
        self.seasonal_period = seasonal_period
        self.n_lags = n_lags or seasonal_period
        if model == "gbrt":
            self.model = GradientBoostingRegressor(random_state=random_state)
        elif model == "rf":
            self.model = RandomForestRegressor(
                random_state=random_state, n_estimators=300
            )
        else:
            raise ValueError(f"unknown model {model!r}")
        self._history = []
        self._next_trading_day = 0

    def _row(self, sales_history, dow, promo, state_holiday, school_holiday, t):
        window = sales_history[-self.n_lags :]
        row = {f"lag_{k}": sales_history[-k] for k in range(1, self.n_lags + 1)}
        row["rolling_mean"] = float(np.mean(window))
        row["rolling_std"] = float(np.std(window))
        row["DayOfWeek"] = dow
        row["Promo"] = promo
        row["StateHolidayFlag"] = int(state_holiday != "none")
        row["SchoolHoliday"] = school_holiday
        row["TradingDay"] = t
        return row

    def fit(self, train: pd.DataFrame) -> "LagFeatureMLForecaster":
        sales = train["Sales"].to_numpy(dtype=float)
        rows, targets = [], []
        for t in range(self.n_lags, len(sales)):
            rows.append(
                self._row(
                    sales[:t],
                    int(train["DayOfWeek"].iloc[t]),
                    int(train["Promo"].iloc[t]),
                    train["StateHoliday"].iloc[t],
                    int(train["SchoolHoliday"].iloc[t]),
                    t,
                )
            )
            targets.append(sales[t])
        X = pd.DataFrame(rows)
        self.model.fit(X, np.array(targets))
        self._history = list(sales)
        self._next_trading_day = len(sales)
        return self

    def predict(self, horizon: int, future: pd.DataFrame) -> np.ndarray:
        history = list(self._history)
        preds = []
        for i in range(horizon):
            row = self._row(
                np.array(history),
                int(future["DayOfWeek"].iloc[i]),
                int(future["Promo"].iloc[i]),
                future["StateHoliday"].iloc[i],
                int(future["SchoolHoliday"].iloc[i]),
                self._next_trading_day + i,
            )
            pred = float(self.model.predict(pd.DataFrame([row]))[0])
            preds.append(pred)
            history.append(pred)
        return np.array(preds)
