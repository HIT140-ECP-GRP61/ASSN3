#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge two cleaned datasets into a single monthly summary file (robust to non-numeric columns).
- Input: cleaned_dataset1.csv, cleaned_dataset2.csv
- Output: monthly_merged.csv
Notes:
  * Aggregates only numeric/bool columns by default to avoid pandas errors.
  * For dataset2, common count columns are summed (if present):
        bat_landing_number, rat_minutes, rat_arrival_number
    while continuous measures are averaged.
"""

import pandas as pd
import numpy as np
import calendar
import re
from pathlib import Path

DATA_DIR = Path(".")
FILE1 = DATA_DIR / "cleaned_dataset1.csv"
FILE2 = DATA_DIR / "cleaned_dataset2.csv"
OUT_CSV = DATA_DIR / "monthly_merged.csv"

def month_to_num(m):
    if pd.isna(m): 
        return np.nan
    s = str(m).strip().lower()
    if re.fullmatch(r"\d{1,2}", s):
        n = int(s)
        return n if 1 <= n <= 12 else np.nan
    month_names = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
    month_abbr  = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
    if s in month_names: return month_names[s]
    if s in month_abbr: return month_abbr[s]
    return np.nan

def aggregate_dataset1(df1: pd.DataFrame) -> pd.DataFrame:
    if "month" not in df1.columns:
        raise ValueError("Dataset1 missing 'month' column.")
    df1 = df1.copy()
    df1["month"] = df1["month"].astype(str).str.strip()

    # numeric/bool columns only (exclude 'month')
    num_cols = df1.select_dtypes(include=["number", "bool"]).columns.tolist()
    num_cols = [c for c in num_cols if c != "month"]

    # mean for numeric/bool
    agg = df1.groupby("month", as_index=False)[num_cols].mean()
    # count of rows per month
    agg["_rows_df1"] = df1.groupby("month").size().values
    return agg

def aggregate_dataset2(df2: pd.DataFrame) -> pd.DataFrame:
    if "month" not in df2.columns:
        raise ValueError("Dataset2 missing 'month' column.")
    df2 = df2.copy()
    df2["month"] = df2["month"].astype(str).str.strip()

    num_cols = df2.select_dtypes(include=["number", "bool"]).columns.tolist()
    num_cols = [c for c in num_cols if c != "month"]

    # Identify common "count" columns to sum
    count_cols = [c for c in ["bat_landing_number","rat_minutes","rat_arrival_number"] if c in df2.columns]
    mean_cols  = [c for c in num_cols if c not in count_cols]

    parts = []
    if mean_cols:
        parts.append(df2.groupby("month", as_index=False)[mean_cols].mean())
    if count_cols:
        parts.append(df2.groupby("month", as_index=False)[count_cols].sum())

    # Combine parts on 'month'
    if len(parts) == 0:
        agg = df2.groupby("month", as_index=False).size()
        agg.rename(columns={"size":"_rows_df2"}, inplace=True)
    else:
        agg = parts[0]
        for p in parts[1:]:
            agg = pd.merge(agg, p, on="month", how="outer")

    # row count per month
    agg["_rows_df2"] = df2.groupby("month").size().values

    return agg

def main():
    df1 = pd.read_csv(FILE1)
    df2 = pd.read_csv(FILE2)

    m1 = aggregate_dataset1(df1)
    m2 = aggregate_dataset2(df2)

    merged = pd.merge(m1, m2, on="month", how="outer")
    merged["_month_num"] = merged["month"].map(month_to_num)
    merged = merged.sort_values(by=["_month_num","month"]).drop(columns=["_month_num"])

    merged.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
