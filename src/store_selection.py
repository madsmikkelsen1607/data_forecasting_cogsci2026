"""Cross-tabulation and stratified selection of a handful of stores.

Selection covers three axes: StoreType, Promo2 (continuous promotions),
and CompetitionDistance (bucketed into near/far by the median). Stores
with incomplete sales history, or with unknown CompetitionDistance (no
competitor on record), are excluded before selection so the chosen stores
have a usable daily series and a meaningful distance tier.
"""

from dataclasses import dataclass

import pandas as pd

EXPECTED_CALENDAR_DAYS = 942  # 2013-01-01 .. 2015-07-31 inclusive
COMPLETENESS_THRESHOLD = 0.95  # fraction of expected days a store must have rows for


def add_distance_tier(store_df: pd.DataFrame) -> pd.DataFrame:
    df = store_df.copy()
    median_dist = df["CompetitionDistance"].median()
    df["DistanceTier"] = pd.cut(
        df["CompetitionDistance"],
        bins=[-1, median_dist, float("inf")],
        labels=["near", "far"],
    )
    return df


def cross_tab(store_df: pd.DataFrame) -> pd.DataFrame:
    df = add_distance_tier(store_df)
    return pd.crosstab(
        [df["StoreType"], df["Promo2"]], df["DistanceTier"], dropna=False
    )


def completeness_by_store(train_df: pd.DataFrame) -> pd.Series:
    counts = train_df.groupby("Store")["Date"].size()
    return counts / EXPECTED_CALENDAR_DAYS


@dataclass
class SelectionResult:
    store_ids: list
    justification: pd.DataFrame


def filter_candidates(
    store_df: pd.DataFrame,
    train_df: pd.DataFrame,
    completeness_threshold: float = COMPLETENESS_THRESHOLD,
) -> pd.DataFrame:
    """Stores with a usable (near-complete) sales history and a known
    CompetitionDistance - the pool select_stores() picks 6 from."""
    df = add_distance_tier(store_df)
    completeness = completeness_by_store(train_df)
    df = df.merge(
        completeness.rename("Completeness"), left_on="Store", right_index=True
    )
    return df[
        (df["Completeness"] >= completeness_threshold)
        & df["CompetitionDistance"].notna()
    ].copy()


def select_stores(
    store_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 6,
    completeness_threshold: float = COMPLETENESS_THRESHOLD,
) -> SelectionResult:
    complete = filter_candidates(store_df, train_df, completeness_threshold)

    complete["Cell"] = list(
        zip(complete["StoreType"], complete["Promo2"], complete["DistanceTier"])
    )
    cell_size = complete["Cell"].value_counts()

    facets = set()
    for st in sorted(complete["StoreType"].dropna().unique()):
        facets.add(("StoreType", st))
    for p2 in (0, 1):
        facets.add(("Promo2", p2))
    for tier in ("near", "far"):
        facets.add(("DistanceTier", tier))

    def facets_of(row) -> set:
        return {
            ("StoreType", row["StoreType"]),
            ("Promo2", row["Promo2"]),
            ("DistanceTier", row["DistanceTier"]),
        }

    selected_ids: list[int] = []
    covered_facets: set = set()
    covered_cells: set = set()
    reasons: list[str] = []

    remaining = complete.copy()
    while len(selected_ids) < n and not remaining.empty:
        remaining["new_facets"] = remaining.apply(
            lambda r: len(facets_of(r) - covered_facets), axis=1
        )
        remaining["new_cell"] = remaining["Cell"].apply(
            lambda c: c not in covered_cells
        )
        # Prefer: most new facets covered, then a brand-new cell combination,
        # then a rarer cell (more distinctive), for a stable tie-break pick
        # the smallest Store id.
        remaining["cell_rarity"] = remaining["Cell"].map(cell_size)
        remaining = remaining.sort_values(
            ["new_facets", "new_cell", "cell_rarity", "Store"],
            ascending=[False, False, True, True],
        )
        pick = remaining.iloc[0]
        selected_ids.append(int(pick["Store"]))

        new_facets = facets_of(pick) - covered_facets
        if new_facets:
            reason = "adds coverage for " + ", ".join(
                f"{k}={v}" for k, v in sorted(new_facets)
            )
        elif pick["Cell"] not in covered_cells:
            reason = (
                f"new StoreType/Promo2/Distance combination "
                f"{pick['StoreType']}/{pick['Promo2']}/{pick['DistanceTier']} "
                f"not yet represented"
            )
        else:
            reason = "additional replicate of an already-covered combination"
        reasons.append(reason)

        covered_facets |= facets_of(pick)
        covered_cells.add(pick["Cell"])
        remaining = remaining[remaining["Store"] != pick["Store"]]

    justification = complete[complete["Store"].isin(selected_ids)][
        [
            "Store",
            "StoreType",
            "Assortment",
            "Promo2",
            "CompetitionDistance",
            "DistanceTier",
            "Completeness",
        ]
    ].copy()
    justification = justification.set_index("Store").loc[selected_ids].reset_index()
    justification["Reason"] = reasons

    return SelectionResult(store_ids=selected_ids, justification=justification)
