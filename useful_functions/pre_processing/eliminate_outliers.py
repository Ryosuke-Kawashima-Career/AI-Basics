import pandas as pd
def calc_zscore_and_remove(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Return a copy of df without rows containing z-score outliers."""
    if threshold <= 0:
        raise ValueError("The threshold must be equal to or greater than zero")
    selected = df.loc[:, columns]
    z_scores = (selected - selected.mean()) / (selected.std(ddof=0) + 1e-5)
    outlier_rows = z_scores.abs().gt(threshold).any(axis=1)
    non_outliers = df.loc[~outlier_rows].copy()
    return non_outliers
