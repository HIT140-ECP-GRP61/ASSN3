import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment
"""
BAI_mult∼C(season)
BAI_mult=β0​+β1​(Summer)+β2​(Winter)+β3​(Spring)+ϵ
while Autumn is the baseline (intercept)
"""
IN_PATH = "engineered_bai.csv"    
BASELINE_SEASON = "Autumn"        
USE_ROBUST_SE = True              

def load_engineered(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. Make sure you ran the feature engineering step "
            f"and saved engineered_bai.csv to this location."
        )
    df = pd.read_csv(path)
    # enforce dtypes 
    if "season" not in df.columns:
        raise KeyError("Column 'season' not found. Did you build season in feature engineering?")
    df["season"] = df["season"].astype("category")
    # ensure the requested baseline exists
    if BASELINE_SEASON in list(df["season"].cat.categories):
        df["season"] = df["season"].cat.reorder_categories(
            ["Summer", "Autumn", "Winter", "Spring"], ordered=False
        )
        
        cats = list(df["season"].cat.categories)
        cats.remove(BASELINE_SEASON)
        df["season"] = df["season"].cat.reorder_categories([BASELINE_SEASON] + cats, ordered=False)
    # helpful downcasts reducing size
    for c in ["bat_landing_to_food", "bat_landing_number", "BAI_mult"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("int8")
    return df


    #TODO: 
    # 1. complete regression BAI_mult=β0​+β1​(Summer)+β2​(Winter)+β3​(Spring)+ϵ

    # 1. write out regression results to a text file
