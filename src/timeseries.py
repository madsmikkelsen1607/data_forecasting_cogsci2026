"""Build a per-store trading-day series from the merged train+store data.

Closed days (Open == 0) are dropped: Sales is a deterministic 0 whenever a
store is closed, so those rows carry no demand signal and would otherwise
zero-inflate the series. The remaining open days are re-indexed by trading
day (0, 1, 2, ...) rather than by calendar date, which keeps a regular grid
for the classical models. Because most stores close on a fixed day (usually
Sunday), the trading-week is regular and its length gives the seasonal
period used by the seasonal-naive and SARIMA models.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class StoreSeries:
    store_id: int
    frame: pd.DataFrame  # trading-day indexed, includes Date/Sales/covariates
    seasonal_period: int


def build_store_series(merged_df: pd.DataFrame, store_id: int) -> StoreSeries:
    store_df = (
        merged_df.loc[merged_df["Store"] == store_id]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    open_df = store_df.loc[store_df["Open"] == 1].reset_index(drop=True)
    open_df.index.name = "TradingDay"

    seasonal_period = infer_seasonal_period(open_df)

    keep_cols = [
        "Date",
        "Sales",
        "Customers",
        "DayOfWeek",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    ]
    frame = open_df[keep_cols].copy()
    return StoreSeries(store_id=store_id, frame=frame, seasonal_period=seasonal_period)


def infer_seasonal_period(open_df: pd.DataFrame) -> int:
    """Typical number of open days per calendar week for this store."""
    n_open_weekdays = open_df["DayOfWeek"].nunique()
    return int(n_open_weekdays)
