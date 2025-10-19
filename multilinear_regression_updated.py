import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

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

def ols_fit(X: np.ndarray, y: np.ndarray):
    #  least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat

    n, p = X.shape
    dof = max(n - p, 1)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) @ (y - y.mean())))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / dof

    sigma2 = ss_res / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.clip(np.diag(var_beta), 0.0, np.inf))

    return beta, se_beta, yhat, resid, r2, adj_r2, dof

def print_table(names, beta, se, z=1.96):
    ci_lo = beta - z * se
    ci_hi = beta + z * se
    header = f"{'Term':<28} {'Coef':>12} {'SE':>12} {'95% CI Low':>14} {'95% CI High':>14}"
    print("\n" + header)
    print("-" * len(header))
    for nm, b, s, lo, hi in zip(names, beta, se, ci_lo, ci_hi):
        print(f"{nm:<28} {b:>12.6f} {s:>12.6f} {lo:>14.6f} {hi:>14.6f}")



def load_engineered(df: pd.DataFrame):
    ap = argparse.ArgumentParser(description="NumPy/Pandas OLS: BAI_scaled ~ C(season)")
    #ap.add_argument("--in", dest="in_path", default=path)
    ap.add_argument("--baseline", default=BASELINE_SEASON)
    args = ap.parse_args()

    #df = pd.read_csv(args.in_path)
    if "season" not in df.columns or "BAI_scaled" not in df.columns:
        raise KeyError("engineered_bai.csv must contain 'season' and 'BAI_scaled'.")

    # Build design and fit
    X, y, names = build_design(df, y_col="BAI_scaled", season_col="season", baseline=args.baseline)
    beta, se, yhat, resid, r2, adj_r2, dof = ols_fit(X, y)

    print("\n====================== OLS: BAI_scaled ~ C(season) (np/pd only) ======================")
    print(f"n = {len(y)}, p = {X.shape[1]}, dof = {dof}")
    print(f"R^2 = {r2:.4f},  Adjusted R^2 = {adj_r2:.4f}")
    print_table(names, beta, se, z=1.96)

    
    desc = (df.groupby("season")["BAI_scaled"]
              .agg(mean="mean", sd="std", n="count")
              .reset_index())
    desc["se"] = desc["sd"] / np.sqrt(desc["n"].clip(lower=1))
    desc["ci95_low"]  = desc["mean"] - 1.96 * desc["se"]
    desc["ci95_high"] = desc["mean"] + 1.96 * desc["se"]
    print("\nSeasonal means of BAI_scaled (descriptive, 95% normal CI):")
    print(desc.to_string(index=False))

    return r2, resid, beta

def visualise_collinearity(df: pd.DataFrame):
    seasonal_dummies = pd.get_dummies(df["season"], prefix='season', drop_first=True, dtype=int)
    df_collinearity = pd.DataFrame(df["BAI_scaled"]).join(seasonal_dummies)
    #Collinearity check
    corr = df_collinearity.corr()
    # Plot the pairwise correlation as heatmap
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=False,
        annot=True
    )

    # customise the labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()
    return

def visualise_raw(path: str):
    df = pd.read_csv(path)
    df["BAI_raw"] = df["bat_landing_to_food"] / df["bat_landing_number"]
    season = df["season"]
    autumn = season == "Autumn"
    winter = season == "Winter"
    summer = season == "Summer"
    
    autumn_bai_values = df[autumn]["BAI_raw"]
    winter_bai_values = df[winter]["BAI_raw"]
    summer_bai_values = df[summer]["BAI_raw"]
    
    x1 = autumn_bai_values.mean()
    x2 = summer_bai_values.mean()
    x3 = winter_bai_values.mean()

    x = [x1, x2, x3]
    labels = ["Autumn", "Summer", "Winter"]

    plt.bar(labels, x, color=['sandybrown', 'lightgreen', 'skyblue'])
    plt.title("Distribution of Average Bat Avoidance by Season")
    plt.ylabel("BAI_raw (Bat Landing Ratio)")
    plt.xlabel("Season")
    
    plt.show()
    
    return df

    #TODO: 
    # 1. write out regression results to a text file

def visualise_transformed(df: pd.DataFrame):
#Visualise Transformed data (log and standardisation)
    df["BAI_LOG"] = df["BAI_raw"].apply(np.log)
    
    Q1 = df["BAI_LOG"].quantile(0.25)
    Q3 = df["BAI_LOG"].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the DataFrame to keep only non-outliers
    non_outliers_mask = (df["BAI_LOG"] >= lower_bound) & (df["BAI_LOG"] <= upper_bound)
    df_cleaned = df[non_outliers_mask].copy()
    
    # Report the change (Crucial feedback)
    n_original = len(df)
    n_cleaned = len(df_cleaned)
    n_removed = n_original - n_cleaned
    
    print(f"\n--- Outlier Removal using IQR on BAI_LOG ---")
    print(f"Original observations (n): {n_original}")
    print(f"Removed outliers (n): {n_removed}")
    print(f"Final observations for OLS (n): {n_cleaned} ({n_removed / n_original * 100:.2f}% removed)")
    
    scaler = StandardScaler()
    Z_BAI_LOG = scaler.fit_transform(df_cleaned[["BAI_LOG"]])
    season = df_cleaned["season"]
    autumn = season == "Autumn"
    winter = season == "Winter"
    summer = season == "Summer"
    df_cleaned["BAI_scaled"] = Z_BAI_LOG
    labels = ["Autumn", "Summer", "Winter"]
    z_autumn_bai_values = df_cleaned.loc[autumn, "BAI_scaled"]
    z_summer_bai_values = df_cleaned.loc[summer, "BAI_scaled"]
    z_winter_bai_values = df_cleaned.loc[winter, "BAI_scaled"]
    
    z_x1 = z_autumn_bai_values.mean()
    z_x2 = z_summer_bai_values.mean()
    z_x3 = z_winter_bai_values.mean()

    z_x = [z_x1, z_x2, z_x3]
   

    plt.bar(labels, z_x, color=['sandybrown', 'lightgreen', 'skyblue'])
    plt.title("Distribution of Average Bat Avoidance by Season")
    plt.ylabel("BAI_scaled (Bat Landing Ratio)")
    plt.xlabel("Season")
    
    plt.show()
    return df_cleaned

def run_base_model(df: pd.DataFrame):
# Temporarily copy BAI_raw to the expected Y column name
    df_base = df.copy()
    df_base["BAI_temp"] = df_base["BAI_raw"] # Use the raw, non-transformed data

    # Re-run arg parsing to get baseline
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=BASELINE_SEASON)
    args, unknown = ap.parse_known_args()

    # Build design matrix (X is the same dummies as before)
    X, y_raw, names = build_design(df_base, y_col="BAI_temp", season_col="season", baseline=args.baseline)
    
    # Fit the OLS model
    beta, se, yhat, resid, r2, adj_r2, dof = ols_fit(X, y_raw)

    print("=============== BASE MODEL: BAI_raw (season) ===============")
    print(f"n = {len(y_raw)}, R^2 = {r2:.4f}, Adjusted R^2 = {adj_r2:.4f}")
    print_table(names, beta, se)

    # Visualize Residuals for the BASE Model (Crucial Check)
    plt.figure(figsize=(8, 4))
    plt.scatter(yhat, resid, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("BASE Model Residuals (BAI_raw) vs. Fitted Values")
    plt.xlabel("Fitted Values (Predicted BAI_raw)")
    plt.ylabel("Residuals (Error)")
    plt.show()

    return r2, resid, beta

def main():
    
    df_raw = visualise_raw(IN_PATH)
    df_transformed = visualise_transformed(df_raw)
    visualise_collinearity(df_transformed)
    r2_base, resid_base, beta_base = run_base_model(df_raw)
    r2_opt, resid_opt, beta_opt = load_engineered(df_transformed)
    
    print("BASE MODEL (BAI_raw) R2 = ", r2_base)
    print("OPT MODEL (BAI_scaled) R2 = ", r2_opt)

main()
