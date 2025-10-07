import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
"""
Feature engineering for Bat Avoidance Index (BAI)
"""
#TODO: 1. engineered dataset is too big, need to reduce size
#      2. add more features? e.g. polynomial terms, etc.
#   3. assume northern hemisphere for season mapping
def season_from_month(m: int) -> str:
    # Southern Hemisphere presume???
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

def build_bai(
    d1_path: str,
    d2_path: str,
    out_path: str,
    k: float = 1.0,
    do_winsor: bool = True,
):
    d1 = pd.read_csv(d1_path)  
    d2 = pd.read_csv(d2_path)  

    # Merge on month inner join 
    df = pd.merge(d1, d2, on="month", how="inner", suffixes=("_bat", "_env"))

  
    core_cols = ["bat_landing_to_food", "bat_landing_number"]

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan) 
    df = df.dropna(subset=core_cols).copy()

    #  dampen extreme outliers
    if do_winsor:
        for c in core_cols:
            df[c] = winsorize_series(df[c], 0.01, 0.99)


    scaler = StandardScaler()
    Z = pd.DataFrame(
        scaler.fit_transform(df[core_cols]),
        columns=[f"Z_{c}" for c in core_cols],
        index=df.index
    )
    df = pd.concat([df, Z], axis=1)

  
    #    Higher BAI_mult = more avoidance 
    #    Stabilizer k prevents division  0
    df["BAI_mult"] = df["Z_bat_landing_to_food"] / (df["Z_bat_landing_number"] + k)


    # Season mapping from month (0–11)
    df["season"] = df["month"].apply(season_from_month).astype("category")
    # save to CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    df.to_csv(out_path, index=False)
    # Preview some columns
    preview_cols = [
        "month", "season",
        "bat_landing_to_food", "bat_landing_number",
        "Z_bat_landing_to_food", "Z_bat_landing_number",
        "BAI_mult"
    ]
    print(df[preview_cols].head(10).to_string(index=False))
    print(f"\nSaved feature engineered dataset -> {out_path}")