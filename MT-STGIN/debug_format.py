#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify data format from get_source_data"""
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, ".")

from models.hyparameter import parameter
from models.data_next import DataClass
import argparse

# Create hyperparameters
para_obj = parameter(argparse.ArgumentParser())
para = para_obj.get_para()

# Load data through DataClass (this calls get_source_data)
print("Loading data...")
data_loader = DataClass(hp=para)

print(f"\nData shape: {data_loader.data_s.shape}")
print(f"Data columns: {list(data_loader.data_s.columns)}")
print(f"\nFirst 10 rows:")
print(data_loader.data_s.head(10))

print(f"\nColumn statistics:")
for col in data_loader.data_s.columns:
    col_data = data_loader.data_s[col].values
    print(
        f"  {col}: min={col_data.min()}, max={col_data.max()}, dtype={col_data.dtype}"
    )

# Check for any values > 23 in the data
print(f"\nChecking for any value >= 24 in all columns:")
for col in data_loader.data_s.columns:
    col_data = data_loader.data_s[col].values
    invalid = np.where(col_data >= 24)[0]
    if len(invalid) > 0:
        print(f"  {col}: Found {len(invalid)} values >= 24")
        print(f"    Examples: {col_data[invalid[:5]]}")
