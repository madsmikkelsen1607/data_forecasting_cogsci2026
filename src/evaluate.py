"""Time-respecting train/validation/test split and forecaster comparison.

The test window is a realistic 6-week-ahead horizon (the kind of lead time
staffing/inventory decisions would actually need), with a preceding 6-week
validation window. Both are measured in trading days (a store's own
open-days-per-week), not calendar days.
"""

from typing import Callable

import numpy as np
import pandas as pd

from timeseries import StoreSeries


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def time_split(
    frame: pd.DataFrame, seasonal_period: int, test_weeks: int = 6, val_weeks: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = test_weeks * seasonal_period
    val_size = val_weeks * seasonal_period
    n = len(frame)
    train_end = n - val_size - test_size
    if train_end <= seasonal_period:
        raise ValueError("not enough history left for training after the split")
    train = frame.iloc[:train_end].reset_index(drop=True)
    val = frame.iloc[train_end : train_end + val_size].reset_index(drop=True)
    test = frame.iloc[train_end + val_size :].reset_index(drop=True)
    return train, val, test


def run_comparison(
    store_series: StoreSeries,
    forecaster_factories: dict[str, Callable[[int], object]],
    test_weeks: int = 6,
    val_weeks: int = 6,
) -> pd.DataFrame:
    frame = store_series.frame
    period = store_series.seasonal_period
    train, val, test = time_split(frame, period, test_weeks, val_weeks)
    train_val = pd.concat([train, val], ignore_index=True)

    rows = []
    for name, make in forecaster_factories.items():
        val_model = make(period).fit(train)
        val_pred = val_model.predict(len(val), val)
        rows.append(
            {
                "store": store_series.store_id,
                "method": name,
                "split": "val",
                "rmse": rmse(val["Sales"].to_numpy(), val_pred),
                "mae": mae(val["Sales"].to_numpy(), val_pred),
            }
        )

        test_model = make(period).fit(train_val)
        test_pred = test_model.predict(len(test), test)
        rows.append(
            {
                "store": store_series.store_id,
                "method": name,
                "split": "test",
                "rmse": rmse(test["Sales"].to_numpy(), test_pred),
                "mae": mae(test["Sales"].to_numpy(), test_pred),
            }
        )

    return pd.DataFrame(rows)


def test_predictions(
    store_series: StoreSeries,
    forecaster_factories: dict[str, Callable[[int], object]],
    test_weeks: int = 6,
    val_weeks: int = 6,
) -> pd.DataFrame:
    """Actual vs. predicted Sales on the test window, one column per method -
    the raw numbers behind run_comparison's test-split metrics, kept here so
    a concrete example forecast can be plotted rather than just scored."""
    frame = store_series.frame
    period = store_series.seasonal_period
    train, val, test = time_split(frame, period, test_weeks, val_weeks)
    train_val = pd.concat([train, val], ignore_index=True)

    out = test[["Date", "Sales"]].copy().rename(columns={"Sales": "Actual"})
    out.insert(0, "store", store_series.store_id)
    out.insert(1, "test_day", range(len(test)))
    for name, make in forecaster_factories.items():
        model = make(period).fit(train_val)
        out[name] = model.predict(len(test), test)
    return out
