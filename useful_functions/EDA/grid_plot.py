import math
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(
    df: pd.DataFrame, 
    explanatory: str, 
    target: str, 
    ax: plt.Axes, 
    target_labels: Optional[dict] = None
):
    """
    Plots the distribution/count of a single explanatory feature split by a binary target onto a given Axes.
    """
    data = df.copy()
    if target_labels:
        data[target] = data[target].map(target_labels).fillna(data[target])
    
    # Automatically switch between histogram (numeric) and count plot (categorical)
    if pd.api.types.is_numeric_dtype(data[explanatory]):
        sns.histplot(
            data=data, 
            x=explanatory, 
            hue=target, 
            kde=True, 
            element="step", 
            ax=ax
        )
        ax.set_title(f"Distribution of {explanatory} by {target}")
    else:
        sns.countplot(
            data=data, 
            x=explanatory, 
            hue=target, 
            ax=ax
        )
        ax.set_title(f"Count of {explanatory} by {target}")
        ax.tick_params(axis='x', rotation=30)
    
    ax.set_xlabel(explanatory)
    ax.set_ylabel("Count")

def plot_grid(
    df: pd.DataFrame, 
    explanatories: List[str], 
    target: str, 
    n_cols: int = 3, 
    target_labels: Optional[dict] = None
):
    """
    Grid-plots the relations of all explanatory features against the target binary values.
    """
    if not explanatories:
        print("No explanatory features provided.")
        return

    n_features = len(explanatories)
    n_rows = math.ceil(n_features / n_cols)
    
    # Create grid figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    
    # Flatten 2D axes array into 1D list for uniform iteration
    if n_features == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
    
    # Plot each explanatory column
    for idx, col_name in enumerate(explanatories):
        plot_histogram(
            df=df, 
            explanatory=col_name, 
            target=target, 
            ax=axes_flat[idx], 
            target_labels=target_labels
        )
    
    # Remove unused empty subplot axes in the bottom row
    for idx in range(n_features, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
        
    plt.tight_layout()
    plt.show()


# --- Runnable Demo ---
if __name__ == "__main__":
    # Create sample customer dataset
    df = pd.DataFrame({
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Month-to-month', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'],
        'Tenure': [1, 24, 60, 3, 12, 48],
        'MonthlyCharges': [70.35, 55.20, 19.85, 85.10, 45.00, 20.00],
        'Churn': [1, 0, 0, 1, 1, 0]
    })
    
    target_col = 'Churn'
    
    # Select explanatory columns (both categorical and continuous) excluding target
    explanatory_cols = [col for col in df.columns if col != target_col]
    
    # Human-readable labels for the binary target
    label_map = {0: "Not Churned", 1: "Churned"}
    
    # Execute plot
    plot_grid(df, explanatories=explanatory_cols, target=target_col, target_labels=label_map)
