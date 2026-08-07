from typing import Optional
import pandas as pd
import numpy as np

def rougher_categorize(
    df: pd.DataFrame, 
    explanatory: str, 
    n_categories: int = 5,
    strategy: str = "quantile",
    encode_as: str = "category",
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Coarsens/bins a continuous numerical feature into discrete categories or bin averages.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    explanatory : str
        Name of the continuous numerical column to coarsen.
    n_categories : int
        Number of bins/quantiles to generate.
    strategy : str
        'quantile' (equal frequency per bin via pd.qcut) or 'uniform' (equal range width via pd.cut).
    encode_as : str
        'category' (Interval categories) or 'mean' (replaces each value with bin average).
    new_col_name : Optional[str]
        New column name. If None, overwrites `explanatory` in place or appends `_binned`.
    """
    # 1. Type validation
    if explanatory not in df.columns:
        raise KeyError(f"Column '{explanatory}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[explanatory]):
        print(f"Skipping '{explanatory}': column is not numeric.")
        return df

    out_col = new_col_name if new_col_name else explanatory

    # 2. Perform Binning
    if strategy == "quantile":
        # Equal-frequency binning (quantile-based)
        bins = pd.qcut(df[explanatory], q=n_categories, duplicates='drop')
    elif strategy == "uniform":
        # Equal-width binning across [min, max]
        bins = pd.cut(df[explanatory], bins=n_categories)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    # 3. Apply Encoding
    if encode_as == "category":
        df[out_col] = bins
    elif encode_as == "mean":
        # Map each sample value to the average feature value of its assigned bin
        bin_means = df[explanatory].groupby(bins, observed=False).transform("mean")
        df[out_col] = bin_means
    else:
        raise ValueError("encode_as must be 'category' or 'mean'")

    return df


# --- Runnable Demo ---
if __name__ == "__main__":
    df_sample = pd.DataFrame({
        'Customer_ID': range(1, 11),
        'MonthlyCharges': [19.85, 25.35, 45.00, 55.20, 70.35, 85.10, 89.90, 95.50, 105.00, 118.75]
    })
    
    # 1. Quantile binning into 3 Interval categories
    df_sample = rougher_categorize(df_sample, explanatory='MonthlyCharges', n_categories=3, encode_as='category', new_col_name='MonthlyCharges_binned')
    
    # 2. Replace values with bin averages
    df_sample = rougher_categorize(df_sample, explanatory='MonthlyCharges', n_categories=3, encode_as='mean', new_col_name='MonthlyCharges_mean')
    
    print(df_sample)
