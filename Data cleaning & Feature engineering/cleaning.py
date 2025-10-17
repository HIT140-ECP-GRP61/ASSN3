# =========================
# HIT140 – Investigation A
# Data Cleaning & Feature Engineering
# =========================

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 0) Config ----------
# Change these if your files live elsewhere
PATH1 = "dataset1.csv"
PATH2 = "dataset2.csv"

OUT1 = "cleaned_dataset1.csv"
OUT2 = "cleaned_dataset2.csv"
REVIEW1 = "review_time_issues_df1.csv"
REVIEW2 = "review_time_issues_df2.csv"

# ---------- 1) Load & basic inspect ----------
df1 = pd.read_csv(PATH1)
df2 = pd.read_csv(PATH2)

print("=== dataset1: shape / head / info ===")
print(df1.shape)
print(df1.head(3))
print(df1.info())

print("\n=== dataset2: shape / head / info ===")
print(df2.shape)
print(df2.head(3))
print(df2.info())

# ---------- 2) Coerce dtypes ----------
# dataset1: timestamps to datetime
time_cols1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
for c in time_cols1:
    if c in df1.columns:
        df1[c] = pd.to_datetime(df1[c], errors="coerce")

# dataset2: observation time to datetime
if "time" in df2.columns:
    df2["time"] = pd.to_datetime(df2["time"], errors="coerce")

# dataset1: numeric columns
num_cols1 = [
    "bat_landing_to_food",
    "seconds_after_rat_arrival",
    "risk",
    "reward",
    "hours_after_sunset",
]
for c in num_cols1:
    if c in df1.columns:
        df1[c] = pd.to_numeric(df1[c], errors="coerce")

# dataset2: numeric columns
num_cols2 = [
    "hours_after_sunset",
    "bat_landing_number",
    "food_availability",
    "rat_minutes",
    "rat_arrival_number",
]
for c in num_cols2:
    if c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

# ---------- 3) Handle missing, duplicates, obvious logic issues ----------

# 3.1 Fill targeted missing (example: habit -> "unknown")
if "habit" in df1.columns:
    df1["habit"] = df1["habit"].fillna("unknown")

# 3.2 Flag rows with any critical time missing (for later manual review)
df1["_missing_time"] = df1[time_cols1].isna().any(axis=1) if time_cols1 else False
df2["_missing_time"] = df2["time"].isna() if "time" in df2.columns else False

# 3.3 Drop exact duplicate rows
before_dupes_1 = df1.shape[0]
df1 = df1.drop_duplicates().reset_index(drop=True)
after_dupes_1 = df1.shape[0]

before_dupes_2 = df2.shape[0]
df2 = df2.drop_duplicates().reset_index(drop=True)
after_dupes_2 = df2.shape[0]

print(f"\nRemoved {before_dupes_1 - after_dupes_1} exact duplicate rows from dataset1.")
print(f"Removed {before_dupes_2 - after_dupes_2} exact duplicate rows from dataset2.")

# 3.4 Validate time logic: rat_period_end must be >= rat_period_start
if set(["rat_period_start", "rat_period_end"]).issubset(df1.columns):
    bad_mask = df1["rat_period_end"] < df1["rat_period_start"]
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        # If you’re confident they’re swapped, you could swap them.
        # To stay conservative/ethical, we set to NaT so duration isn’t computed from wrong order.
        df1.loc[bad_mask, ["rat_period_start", "rat_period_end"]] = np.nan
        print(f"Time logic fixed: set NaT for {bad_count} rows where end < start in dataset1.")

# ---------- 4) Feature Engineering for Investigation A ----------

# 4.1 Rat presence duration in seconds (non-negative)
if set(["rat_period_start", "rat_period_end"]).issubset(df1.columns):
    duration = (df1["rat_period_end"] - df1["rat_period_start"]).dt.total_seconds()
    df1["rat_presence_duration"] = duration.clip(lower=0)
else:
    df1["rat_presence_duration"] = np.nan

# 4.2 rat_present flag at the exact bat landing instant
# True if start_time falls within [rat_period_start, rat_period_end]
if set(["start_time", "rat_period_start", "rat_period_end"]).issubset(df1.columns):
    df1["rat_present"] = (
        (df1["start_time"] >= df1["rat_period_start"]) &
        (df1["start_time"] <= df1["rat_period_end"])
    ).astype(int)
else:
    df1["rat_present"] = np.nan

# 4.3 arrival_delay_bin from seconds_after_rat_arrival (bins are adjustable)
bins = [-np.inf, 0, 30, 120, np.inf]
labels = ["≤0s", "1–30s", "31–120s", ">120s"]
if "seconds_after_rat_arrival" in df1.columns:
    df1["arrival_delay_bin"] = pd.cut(df1["seconds_after_rat_arrival"], bins=bins, labels=labels)
else:
    df1["arrival_delay_bin"] = pd.Series(pd.Categorical([]))

# ---------- 5) IQR outlier flags (for review, do NOT drop automatically) ----------

def flag_iqr_outlier(series: pd.Series) -> pd.Series:
    """Return 1 if the value is an IQR outlier, else 0. Ignores NaNs."""
    s = series.dropna()
    if s.empty:
        return pd.Series([0]*len(series), index=series.index)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ((series < lo) | (series > hi)).fillna(0).astype(int)

out_cols1 = [
    "bat_landing_to_food",
    "seconds_after_rat_arrival",
    "hours_after_sunset",
    "rat_presence_duration",
]
for c in out_cols1:
    if c in df1.columns and pd.api.types.is_numeric_dtype(df1[c]):
        df1[f"out_{c}"] = flag_iqr_outlier(df1[c])

out_cols2 = [
    "hours_after_sunset",
    "bat_landing_number",
    "food_availability",
    "rat_minutes",
    "rat_arrival_number",
]
for c in out_cols2:
    if c in df2.columns and pd.api.types.is_numeric_dtype(df2[c]):
        df2[f"out_{c}"] = flag_iqr_outlier(df2[c])

# ---------- 6) Post-clean QA summaries ----------
print("\n=== Missing values (dataset1) ===")
print(df1.isna().sum().sort_values(ascending=False).head(15))

print("\n=== Outlier flags count (dataset1) ===")
print(df1.filter(like="out_").sum().sort_values(ascending=False))

print("\n=== Missing values (dataset2) ===")
print(df2.isna().sum().sort_values(ascending=False).head(15))

print("\n=== Outlier flags count (dataset2) ===")
print(df2.filter(like="out_").sum().sort_values(ascending=False))

# ---------- 7) Save cleaned outputs + review logs ----------
# Rows with critical missing time → manual review
issues_df1 = df1[df1["_missing_time"] == True].copy()
issues_df2 = df2[df2["_missing_time"] == True].copy()

df1.to_csv(OUT1, index=False)
df2.to_csv(OUT2, index=False)
issues_df1.to_csv(REVIEW1, index=False)
issues_df2.to_csv(REVIEW2, index=False)

print(f"\nSaved cleaned files:\n- {Path(OUT1).resolve()}\n- {Path(OUT2).resolve()}")
print(f"Saved review logs (rows needing manual time review):\n- {Path(REVIEW1).resolve()}\n- {Path(REVIEW2).resolve()}")

# ---------- 8) (Optional) Quick sanity checks to guide Descriptive/Inferential ----------
if "risk" in df1.columns:
    print("\nQuick check – risk-taking rate overall:", round(df1["risk"].mean(), 3))
if {"risk", "rat_present"}.issubset(df1.columns):
    ct = pd.crosstab(df1["rat_present"], df1["risk"], normalize="index")
    print("\nRisk-taking by rat_present (row-normalized):\n", ct)
