"""Loading and light cleaning of the raw Rossmann CSVs."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATE_HOLIDAY_LABELS = {
    "0": "none",
    "a": "public",
    "b": "easter",
    "c": "christmas",
}


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(
        data_dir / "train.csv",
        dtype={
            "Store": "int32",
            "DayOfWeek": "int8",
            "Sales": "int32",
            "Customers": "int32",
            "Open": "int8",
            "Promo": "int8",
            "StateHoliday": "string",
            "SchoolHoliday": "int8",
        },
        parse_dates=["Date"],
    )
    df["StateHoliday"] = (
        df["StateHoliday"].map(STATE_HOLIDAY_LABELS).astype("category")
    )
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def load_store(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(
        data_dir / "store.csv",
        dtype={
            "Store": "int32",
            "StoreType": "category",
            "Assortment": "category",
            "CompetitionDistance": "float64",
            "CompetitionOpenSinceMonth": "float64",
            "CompetitionOpenSinceYear": "float64",
            "Promo2": "int8",
            "Promo2SinceWeek": "float64",
            "Promo2SinceYear": "float64",
            "PromoInterval": "string",
        },
    )
    return df


def load_merged(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Train rows joined with static store metadata."""
    train = load_train(data_dir)
    store = load_store(data_dir)
    return train.merge(store, on="Store", how="left", validate="many_to_one")


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    missing = df.isna().sum()
    return (
        pd.DataFrame({"n_missing": missing, "pct_missing": missing / n * 100})
        .query("n_missing > 0")
        .sort_values("pct_missing", ascending=False)
    )
