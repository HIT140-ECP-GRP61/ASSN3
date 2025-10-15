import argparse
import pandas as pd
import numpy as np
from ols_utils import ols_fit  

"""
BAI_mult∼C(season)
BAI_mult=β0​+β1​(Summer)+β2​(Winter)+β3​(Spring)+ϵ
while Autumn is the baseline (intercept)
"""
IN_PATH = "engineered_bai.csv"    
BASELINE_SEASON = "Autumn"        
USE_ROBUST_SE = True              

def reorder_with_baseline(cats, baseline):
    cats = list(cats)
    if baseline not in cats:
        raise ValueError(f"Baseline '{baseline}' not in categories: {cats}")
    return [baseline] + [c for c in cats if c != baseline]

def build_design(df: pd.DataFrame, y_col: str, season_col: str, baseline: str):
    y = df[y_col].to_numpy(dtype=float)

    # Intercept
    X_parts = [np.ones((len(df), 1))]
    names = ["Intercept"]

    season = df[season_col].astype("category")
    ordered = reorder_with_baseline(season.cat.categories, baseline)
    season = season.cat.reorder_categories(ordered, ordered=False)

    dummies = pd.get_dummies(season, prefix=season_col)
    # first column corresponds to baseline drop it
    keep = dummies.columns[1:]
    X_parts.append(dummies[keep].to_numpy(dtype=float))
    names.extend(list(keep))

    X = np.hstack(X_parts)
    return X, y, names

def print_table(names, beta, se, z=1.96):
    ci_lo = beta - z * se
    ci_hi = beta + z * se
    header = f"{'Term':<28} {'Coef':>12} {'SE':>12} {'95% CI Low':>14} {'95% CI High':>14}"
    print("\n" + header)
    print("-" * len(header))
    for nm, b, s, lo, hi in zip(names, beta, se, ci_lo, ci_hi):
        print(f"{nm:<28} {b:>12.6f} {s:>12.6f} {lo:>14.6f} {hi:>14.6f}")



def load_engineered(path: str):
    ap = argparse.ArgumentParser(description="NumPy/Pandas OLS: BAI_mult ~ C(season)")
    ap.add_argument("--in", dest="in_path", default=path)
    ap.add_argument("--baseline", default=BASELINE_SEASON)
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if "season" not in df.columns or "BAI_mult" not in df.columns:
        raise KeyError("engineered_bai.csv must contain 'season' and 'BAI_mult'.")

    # Build design and fit
    X, y, names = build_design(df, y_col="BAI_mult", season_col="season", baseline=args.baseline)
    beta, se, yhat, resid, r2, adj_r2, dof = ols_fit(X, y)  
    print("\n====================== OLS: BAI_mult ~ C(season) (np/pd only) ======================")
    print(f"n = {len(y)}, p = {X.shape[1]}, dof = {dof}")
    print(f"R^2 = {r2:.4f},  Adjusted R^2 = {adj_r2:.4f}")
    print_table(names, beta, se, z=1.96)

    
    desc = (df.groupby("season")["BAI_mult"]
              .agg(mean="mean", sd="std", n="count")
              .reset_index())
    desc["se"] = desc["sd"] / np.sqrt(desc["n"].clip(lower=1))
    desc["ci95_low"]  = desc["mean"] - 1.96 * desc["se"]
    desc["ci95_high"] = desc["mean"] + 1.96 * desc["se"]
    print("\nSeasonal means of BAI_mult (descriptive, 95% normal CI):")
    print(desc.to_string(index=False))


    #TODO: 
    # 1. complete regression BAI_mult=β0​+β1​(Summer)+β2​(Winter)+β3​(Spring)+ϵ

    # 1. write out regression results to a text file

def main():
    load_engineered(IN_PATH)


main()