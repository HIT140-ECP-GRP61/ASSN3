
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def run_tree_on_df(df, feature_cols, target_col="landing_efficiency", max_depth=4, min_samples_leaf=1, cv_min_n=20, test_size=0.3):
    required = feature_cols + [target_col]
    df2 = df.dropna(subset=required).copy()
    print(f"Rows: {len(df)} | Dropped NA: {len(df) - len(df2)} | Used: {len(df2)}")
    if len(df2) < 3:
        raise ValueError("Need at least 3 rows after cleaning.")
        # Build X (features) and y (target) as numeric arrays for sklearn

    X = df2[feature_cols].to_numpy(dtype=float)
    y = df2[target_col].to_numpy(dtype=float)
        # Create the decision tree regressor 

    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    if len(df2) < cv_min_n:
                # use between 2 and 5 folds depending on sample size

        folds = max(2, min(5, len(df2)))
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(tree, X, y, cv=cv, scoring="r2")
        print(f"CV folds={folds} | CV R^2 mean={scores.mean():.4f} ± {scores.std():.4f}")
        #fit
        tree.fit(X, y)
        y_pred = tree.predict(X)
        print(f"In-sample R^2 (fit on all) = {r2_score(y, y_pred):.4f}")
    else:
        # Holdout evaluation
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
        tree.fit(X_tr, y_tr)
        y_pred = tree.predict(X_te)
        print(f"Holdout R^2={r2_score(y_te, y_pred):.4f} | RMSE={mean_squared_error(y_te, y_pred, squared=False):.4f}")
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_cols, tree.feature_importances_), key=lambda t: -t[1]):
        print(f"{name:30s} {imp:.4f}")
    def d_tree(tree, feature_names, node_id=0, depth=0):
        left = tree.tree_.children_left[node_id]
        right = tree.tree_.children_right[node_id]
        indent = "  " * depth
        if left == -1 and right == -1:
            n_node_samples = tree.tree_.n_node_samples[node_id]
            value = tree.tree_.value[node_id][0][0]
            print(f"{indent}Leaf id={node_id} | n={n_node_samples} | mean={value:.4f}")
        else:
            feat_idx = tree.tree_.feature[node_id]
            thresh = tree.tree_.threshold[node_id]
            feat_name = feature_names[feat_idx]
            print(f"{indent}[ {feat_name} <= {thresh:.4f} ]")
            d_tree(tree, feature_names, left, depth + 1)
            d_tree(tree, feature_names, right, depth + 1)
    print("\n===== Decision Tree =====")
    d_tree(tree, feature_cols)
    return tree

path = "/mnt/data/monthly_features.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    features = ["month", "hours_after_sunset_mean_df1", "risk_x_rat", "reward_to_risk", "rat_time_ratio"]
    if not all(c in df.columns for c in features + ["landing_efficiency"]):
        use_synth = True
    else:
        use_synth = False
else:
    use_synth = True

if use_synth:
    rng = np.random.default_rng(42)
    n = 60
    month = rng.integers(1, 13, size=n)
    hours = rng.normal(3.0, 1.0, size=n).clip(0, 6)
    risk = rng.beta(2, 5, size=n)
    reward_to_risk = rng.normal(1.0, 0.3, size=n).clip(0.1, 2.0)
    rat_time_ratio = rng.uniform(0, 1, size=n)
    seasonal = 0.2 * np.cos(2 * np.pi * (month - 6) / 12) + 0.2
    y = (0.6 + seasonal + 0.1 * hours - 0.5 * risk + 0.05 * reward_to_risk - 0.3 * rat_time_ratio
         + rng.normal(0, 0.05, size=n))
    df = pd.DataFrame({
        "month": month,
        "hours_after_sunset_mean_df1": hours,
        "risk_x_rat": risk,
        "reward_to_risk": reward_to_risk,
        "rat_time_ratio": rat_time_ratio,
        "landing_efficiency": y
    })
    features = ["month", "hours_after_sunset_mean_df1", "risk_x_rat", "reward_to_risk", "rat_time_ratio"]

tree = run_tree_on_df(df, features)

