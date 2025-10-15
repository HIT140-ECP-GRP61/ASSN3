import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
    
"""
Feature engineering for Bat Avoidance Index (BAI) 
 Compress dataset2 to month-level mean landing frequency
Merge on month (1-to-1)
 Multiplicative BAI: Z(delay) / (Z(monthly_landing) + k)
"""

#TODO: 
#      2. add more features? e.g. polynomial terms, etc.
#   3. assume northern hemisphere for season mapping

def season_from_month(m: int) -> str:
    # Southern Hemisphere (adjust 
    if m in [11, 0, 1]:
        return "Summer"
    elif m in [2, 3, 4]:
        return "Autumn"
    elif m in [5, 6, 7]:
        return "Winter"
    else:
        return "Spring"

def winsorize_series(s: pd.Series, p_low=0.01, p_high=0.99) -> pd.Series:
    lo, hi = s.quantile([p_low, p_high])
    return s.clip(lower=lo, upper=hi)

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def build_bai(
    d1_path: str,
    d2_path: str,
    out_path: str,
    k: float = 1.0,
    do_winsor: bool = True,
):
  
    d1 = pd.read_csv(d1_path)   
    d2 = pd.read_csv(d2_path)   #TODO: check columns exist

    # Compress dataset2  month-level mean landing frequency
    '''
    
ctx (monthly mean)

| month | bat_landing_number_month    |
+-------+-----------------------------+
|   1   |           mean              |
|   2   |           mean              |
    
    '''
    ctx = (d2.groupby("month", as_index=False)
             .agg(bat_landing_number_month=("bat_landing_number", "mean")))

    # Merge 1-to-1 on month 
    df = pd.merge(d1, ctx, on="month", how="inner")
    '''
   

df = merge(d1, ctx, on="month")
| month | bat_landing_to_food    | bat_landing_number_month    |
+-------+------------------------+-----------------------------+
|   1   |         ...            |           ...               |
|   2   |         ...            |           ...               |
    
    '''

    #  Clean  winsorize core variables
    
    core_cols = ["bat_landing_to_food", "bat_landing_number_month"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=core_cols).copy()

    if do_winsor:
        for c in core_cols:
            df[c] = winsorize_series(df[c], 0.01, 0.99)

    
    df["Z_bat_landing_to_food"]    = zscore(df["bat_landing_to_food"])
    df["Z_bat_landing_number_mon"] = zscore(df["bat_landing_number_month"])

   
    #    Higher BAI = more avoidance (longer delay, fewer landings)
    df["BAI_mult"] = df["Z_bat_landing_to_food"] / (df["Z_bat_landing_number_mon"] + k)

    # Season category
    df["season"] = df["month"].apply(season_from_month).astype("category")
    '''
    df
| month | bat_landing_to_food    | bat_landing_number_month    | Z_bat_landing_to_food    | Z_bat_landing_number_mon | BAI_mult |
+-------+------------------------+-----------------------------+--------------------------+--------------------------+----------+
|   1   |         ...            |           ...               |           ...            |           ...            |   ...    |
|   2   |         ...            |           ...               |           ...            |           ...            |   ...    |
    '''

    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # Preview
    preview_cols = [
        "month", "season",
        "bat_landing_to_food", "bat_landing_number_month",
        "Z_bat_landing_to_food", "Z_bat_landing_number_mon",
        "BAI_mult"
    ]
    print(df[preview_cols].head(10).to_string(index=False))
    print(f"\nSaved feature engineered dataset -> {out_path}")

