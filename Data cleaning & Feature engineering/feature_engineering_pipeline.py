#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIT140 – Investigation A & B
Feature Engineering Pipeline
Author: Duong Quy Vu (Dave)
---------------------------------------------
Input:
    cleaned_dataset1.csv
    cleaned_dataset2.csv
Output:
    monthly_merged.csv
    monthly_features.csv
"""

import pandas as pd
import numpy as np
import calendar
import re
from pathlib import Path

# ---------- CONFIG ----------
DATA_DIR = Path(".")
DF1_PATH = DATA_DIR / "cleaned_dataset1.csv"
DF2_PATH = DATA_DIR / "cleaned_dataset2.csv"
OUT_MERGED = DATA_DIR / "monthly_merged.csv"
OUT_FEATURES = DATA_DIR / "monthly_features.csv"

# ---------- LOAD ----------
df1 = pd.read_csv(DF1_PATH)
df2 = pd.read_csv(DF2_PATH)

# Ensure month column
for df in (df1, df2):
    if "month" not in df.columns:
        raise ValueError("Expected 'month' column in both datasets.")
    df["month"] = df["month"].astype(str).str.strip()

# ---------- AGGREGATE BY MONTH ----------
agg1 = {
    "risk": "mean",
    "reward": "mean",
    "bat_landing_to_food": "mean",
    "seconds_after_rat_arrival": "mean",
    "hours_after_sunset": "mean",
    "rat_presence_duration": "mean",
    "rat_present": "mean",
}
monthly1 = df1.groupby("month", as_index=False).agg({k:v for k,v in agg1.items() if k in df1.columns})
monthly1.rename(columns={"hours_after_sunset": "hours_after_sunset_mean_df1"}, inplace=True)

agg2 = {
    "bat_landing_number": "sum",
    "rat_minutes": "sum",
    "rat_arrival_number": "sum",
    "food_availability": "mean",
    "hours_after_sunset": "mean",
}
monthly2 = df2.groupby("month", as_index=False).agg({k:v for k,v in agg2.items() if k in df2.columns})
monthly2.rename(columns={"hours_after_sunset": "hours_after_sunset_mean_df2"}, inplace=True)

# ---------- MERGE ----------
merged = pd.merge(monthly1, monthly2, on="month", how="outer")

# ---------- SORT MONTHS ----------
month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
month_abbr = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
def month_to_num(m):
    s = str(m).strip().lower()
    if re.fullmatch(r"\d{1,2}", s):
        n = int(s)
        return n if 1 <= n <= 12 else None
    if s in month_names: return month_names[s]
    if s in month_abbr: return month_abbr[s]
    return None
merged["_month_num"] = merged["month"].map(month_to_num)
merged = merged.sort_values(by=["_month_num", "month"]).drop(columns=["_month_num"])
merged.to_csv(OUT_MERGED, index=False)
print(f"✅ Saved monthly summary → {OUT_MERGED}")

# ---------- FEATURE ENGINEERING ----------
df = merged.copy()
df["month_num"] = df["month"].map(month_to_num)

# Cyclical encoding
angle = 2*np.pi*(df["month_num"]-1)/12.0
df["season_sin"] = np.sin(angle)
df["season_cos"] = np.cos(angle)

# Ratios and derived measures
df["reward_to_risk"] = np.where(df["risk"]!=0, df["reward"]/df["risk"], np.nan)
df["landing_efficiency"] = np.where(df["rat_arrival_number"]!=0, df["bat_landing_number"]/df["rat_arrival_number"], np.nan)
df["rat_time_ratio"] = np.where(df["rat_minutes"]!=0, df["rat_minutes"]/(30*len(df2)), np.nan)
df["landing_per_food"] = np.where(df["food_availability"]!=0, df["bat_landing_number"]/df["food_availability"], np.nan)

# Sunset alignment
df["sunset_alignment_diff"] = df["hours_after_sunset_mean_df1"] - df["hours_after_sunset_mean_df2"]

# Interaction terms
df["reward_x_rat"] = df["reward"] * df["rat_time_ratio"]
df["risk_x_rat"] = df["risk"] * df["rat_time_ratio"]

# Rolling 3-month means and z-scores
df = df.sort_values(["month_num","month"]).reset_index(drop=True)
for col in ["landing_efficiency","rat_time_ratio","landing_per_food"]:
    if col in df.columns:
        df[f"{col}_roll3_mean"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# ---------- EXPORT ----------
df.to_csv(OUT_FEATURES, index=False)
print(f"✅ Saved engineered features → {OUT_FEATURES}")
print("🎯 Feature Engineering completed successfully.")