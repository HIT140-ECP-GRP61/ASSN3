# =========================
# HIT140 – Investigation A
# CLEANING QUALITY CHECKS
# =========================

import pandas as pd

# Load the cleaned outputs you just created
df1 = pd.read_csv("cleaned_dataset1.csv")
df2 = pd.read_csv("cleaned_dataset2.csv")

print("=== 1) Structure & missing values ===")
print("\nDataset1 info:")
print(df1.info())
print("\nMissing (df1):\n", df1.isna().sum().sort_values(ascending=False).head(20))

print("\nDataset2 info:")
print(df2.info())
print("\nMissing (df2):\n", df2.isna().sum().sort_values(ascending=False).head(20))

print("\n=== 2) Duplicates ===")
print("Duplicate rows in df1:", df1.duplicated().sum())
print("Duplicate rows in df2:", df2.duplicated().sum())

print("\n=== 3) Time logic sanity ===")
# If these columns exist, check basic logic
if set(["rat_period_start", "rat_period_end"]).issubset(df1.columns):
    # after cleaning, these should be 0 (we set invalid to NaN)
    bad_logic = (pd.to_datetime(df1["rat_period_end"], errors="coerce") < 
                 pd.to_datetime(df1["rat_period_start"], errors="coerce")).sum()
    print("Rows with rat_period_end < rat_period_start:", bad_logic)
else:
    print("Time columns not present for logic check (dataset1).")

if "start_time" in df1.columns and "sunset_time" in df1.columns:
    # optional: how many bat landings recorded before sunset?
    before_sunset = (pd.to_datetime(df1["start_time"], errors="coerce") <
                     pd.to_datetime(df1["sunset_time"], errors="coerce")).sum()
    print("Rows where bat landed before sunset:", before_sunset)

print("\n=== 4) Outlier flags (counts) ===")
out1 = df1.filter(like="out_").sum().sort_values(ascending=False)
out2 = df2.filter(like="out_").sum().sort_values(ascending=False)
print("Dataset1 outlier flags:\n", out1 if len(out1) else "No outlier flags present.")
print("\nDataset2 outlier flags:\n", out2 if len(out2) else "No outlier flags present.")

print("\n=== 5) Engineered features sanity ===")
cols = [c for c in ["rat_present","rat_presence_duration","arrival_delay_bin"] if c in df1.columns]
if cols:
    print(df1[cols].head(10))
    if "rat_presence_duration" in df1.columns:
        neg_dur = (df1["rat_presence_duration"] < 0).sum()
        print("Negative durations (should be 0):", neg_dur)
else:
    print("Engineered feature columns not found in dataset1.")

print("\n=== 6) Quick descriptive sanity ===")
if "risk" in df1.columns:
    print("Risk-taking rate (mean of risk):", round(pd.to_numeric(df1["risk"], errors="coerce").mean(), 4))
if "bat_landing_to_food" in df1.columns:
    print("Mean time to food:", round(pd.to_numeric(df1["bat_landing_to_food"], errors="coerce").mean(), 2))
if "rat_minutes" in df2.columns:
    print("Mean rat minutes per interval:", round(pd.to_numeric(df2["rat_minutes"], errors="coerce").mean(), 2))

print("\n=== Done: If duplicates==0, bad_logic==0, negative durations==0, and missing only in non-critical columns, your data is ready for analysis. ===")