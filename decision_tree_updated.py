import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
"""
Decision Tree on engineered_bai.csv
Model: BAI_mult ~ month_num + hours_after_sunset

Parses start_time and sunset_time (day-first)
Derives month_num and hours_after_sunset
Uses holdout split for large datasets, CV for small ones
"""

def month_to_season_3month(m: int) -> str:
    """
    Map month number (1-12) to 3-month season (assuming Northern hemisphere now).
    ASCII visualization of mapping:
      Dec(12), Jan(1), Feb(2) -> Winter
      Mar(3), Apr(4), May(5) -> Spring
      Jun(6), Jul(7), Aug(8) -> Summer
      Sep(9), Oct(10), Nov(11) -> Autumn
    """
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"

def _ascii_tree_printer(tree, feature_names):
    
    t = tree.tree_
    def _node(node_id=0, depth=0):
        left = t.children_left[node_id]
        right = t.children_right[node_id]
        indent = "  " * depth
        if left == -1 and right == -1:
            n = t.n_node_samples[node_id]
            val = float(t.value[node_id].ravel().mean())
            print(f"{indent}Leaf id={node_id} | n={n} | mean={val:.4f}")
        else:
            feat = t.feature[node_id]
            thresh = t.threshold[node_id]
            fname = feature_names[feat] if (0 <= feat < len(feature_names)) else f"f{feat}"
            print(f"{indent}[ {fname} <= {thresh:.4f} ]")
            _node(left, depth + 1)
            print(f"{indent}else:")
            _node(right, depth + 1)
    _node(0, 0)

def analyze_season_relevance(df, target="BAI_mult", hours_col="hours_after_sunset",
                             min_samples_leaf=5, max_depth=3, cv_folds=5):
    """
  Sample results:
    Print ASCII tree for combined model and a succinct conclusion:
    Holdout R^2=0.0072 | RMSE=78.0901 | n=466913
    Feature importances:
    hours_after_sunset   0.9068
    month_num            0.0932
    """
    if "month_num" not in df.columns:
        raise KeyError("month_num required to derive season (map from month).")

    df2 = df.copy()
    df2["season"] = df2["month_num"].apply(lambda x: month_to_season_3month(int(x)) if not pd.isna(x) else np.nan)
    # drop rows 
    df2 = df2.dropna(subset=[target, "season", hours_col]).copy()
    if len(df2) < 3:
        print("Not enough rows to analyze season relevance (need >=3).")
        return

    try:
        enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    X_season = enc.fit_transform(df2[["season"]])
    season_names = list(enc.get_feature_names_out(["season"]))

    # numeric hours feature
    X_hours = df2[[hours_col]].to_numpy(dtype=float)

    # combined
    X_both = np.hstack([X_season, X_hours])
    feature_names = season_names + [hours_col]

    y = df2[target].to_numpy(dtype=float)

    # small tree models and cross-validated R^2 where possible
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    def cv_r2(X, y):
        folds = min(cv_folds, max(2, len(y)))
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(tree, X, y, cv=cv, scoring="r2")
        return scores.mean(), scores.std()

    r2_season_mean, r2_season_std = cv_r2(X_season, y)
    r2_hours_mean, r2_hours_std = cv_r2(X_hours, y)
    r2_both_mean, r2_both_std = cv_r2(X_both, y)

    # fit final combined tree for inspection
    tree.fit(X_both, y)
    importances = tree.feature_importances_
    season_importance = float(sum(importances[: len(season_names)]))

    # Print  
    print("\n=== Season relevance diagnostics (3-month seasons) ===")
    print(f"n = {len(df2)} rows used")
    print(f"Cross-validated R^2 (season-only) = {r2_season_mean:.4f} ± {r2_season_std:.4f}")
    print(f"Cross-validated R^2 (hours-only)  = {r2_hours_mean:.4f} ± {r2_hours_std:.4f}")
    print(f"Cross-validated R^2 (season+hours) = {r2_both_mean:.4f} ± {r2_both_std:.4f}")
    print("\nFeature importances (combined model):")
    for name, imp in zip(feature_names, importances):
        print(f"  {name:25s} {imp:.4f}")
    print("\nASCII decision tree (combined model):")
    _ascii_tree_printer(tree, feature_names)

    # Intuitive conclusion
    conclude = False
    reasons = []
    if r2_season_mean > r2_hours_mean:
        conclude = True
        reasons.append("season-only R^2 > hours-only R^2")
    if season_importance >= 0.10:
        conclude = True
        reasons.append(f"season features explain {season_importance:.2%} of importance (>=10%)")

    if conclude:
        print("\nCONCLUSION: BAI appears related to season (3-month grouping).")
        print("Reasons: " + "; ".join(reasons))
    else:
        print("\nCONCLUSION: No strong evidence that BAI is related to season (3-month grouping).")
        print("Hints: increase data, check aggregation, or use other feature encodings if needed.")

def main():
    ap = argparse.ArgumentParser(description="Decision Tree: BAI_mult ~ month + hours_after_sunset")
    ap.add_argument("--in", dest="in_path", default="engineered_bai.csv")
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--min_samples_leaf", type=int, default=20)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--cv_min_n", type=int, default=30000)
    ap.add_argument("--analyze_season", action="store_true", help="Run season relevance analysis and print tree/conclusion")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)

    # Parse datetimes 
    for col in ["start_time", "sunset_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # month feature
    if "start_time" in df.columns and df["start_time"].notna().any():
        df["month_num"] = df["start_time"].dt.month
    elif "month" in df.columns:
        df["month_num"] = pd.to_numeric(df["month"], errors="coerce")
    else:
        raise KeyError("No 'start_time' or 'month' column to derive month_num.")

    # hours_after_sunset feature
    if "start_time" in df.columns and "sunset_time" in df.columns:
        df["hours_after_sunset"] = (df["start_time"] - df["sunset_time"]).dt.total_seconds() / 3600.0
    else:
        raise KeyError("Need both 'start_time' and 'sunset_time' to derive hours_after_sunset.")

    target = "BAI_mult"
    features = ["month_num", "hours_after_sunset"]
    df2 = df.dropna(subset=features + [target]).copy()

    # run season analysis
    if args.analyze_season:
        try:
            analyze_season_relevance(df, target=target, hours_col="hours_after_sunset",
                                     min_samples_leaf=args.min_samples_leaf, max_depth=args.max_depth)
        except Exception as e:
            print("Season analysis failed:", e)

    X = df2[features].to_numpy(dtype=float)
    y = df2[target].to_numpy(dtype=float)

    n = len(df2)
    if n < args.cv_min_n:
        # cross-validation for small/medium sets
        folds = 5 if n >= 5 else max(2, n)
        tree = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, random_state=42)
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(tree, X, y, cv=cv, scoring="r2")
        print(f"CV folds={folds} | CV R^2 mean={scores.mean():.4f} ± {scores.std():.4f}")
        tree.fit(X, y)
    else:
        # holdout for very large sets
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=42)
        tree = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, random_state=42)
        tree.fit(X_tr, y_tr)
        y_pred = tree.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        try:
            rmse = mean_squared_error(y_te, y_pred, squared=False)
        except TypeError:
            # compute RMSE manually
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        print(f"Holdout R^2={r2:.4f} | RMSE={rmse:.4f} | n={n}")

    print("Feature importances:")
    for name, imp in sorted(zip(features, tree.feature_importances_), key=lambda t: -t[1]):
        print(f"{name:20s} {imp:.4f}")

if __name__ == "__main__":
    main()