#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check what minute values we're getting"""
import pandas as pd
import numpy as np

file_path = "../3S-TBLN/data/METR-LA/train.csv"
data = pd.read_csv(file_path, encoding="utf-8")
data["date"] = pd.to_datetime(data["date"])

# METR-LA has 5-minute intervals
minute_values = data["date"].dt.minute.values

print(f"Raw minute values in METR-LA:")
print(f"  Unique values: {sorted(np.unique(minute_values))}")
print(f"  Min: {minute_values.min()}, Max: {minute_values.max()}")

print(f"\nMinute // 15 (proposed for vocab_size=4):")
quantized_15 = minute_values // 15
print(f"  Unique values: {sorted(np.unique(quantized_15))}")
print(f"  Min: {quantized_15.min()}, Max: {quantized_15.max()}")

print(f"\nMinute // 5 (for 12 bins):")
quantized_5 = minute_values // 5
print(f"  Unique values: {sorted(np.unique(quantized_5))}")
print(f"  Min: {quantized_5.min()}, Max: {quantized_5.max()}")

print(f"\nIf we accidentally used raw minute values:")
print(f"  Min: {minute_values.min()}, Max: {minute_values.max()}")
print(f"  Unique values: {sorted(np.unique(minute_values))}")
