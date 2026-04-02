#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug data transformation to find where hour=25 comes from"""
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

print(f"Parameters:")
print(f"  site_num: {para.site_num}")
print(f"  input_length: {para.input_length}")
print(f"  output_length: {para.output_length}")
print(f"  batch_size: {para.batch_size}")
print(f"  file_train_s: {para.file_train_s}")

# Load data through DataClass
data_loader = DataClass(hp=para)

print(f"\nData loaded:")
print(f"  Data shape: {data_loader.data_s.shape}")
print(f"  Data length: {data_loader.length}")

# Check hour values in loaded data
hour_data = data_loader.data_s["hour"].values
print(f"\nHour column statistics:")
print(f"  Min: {hour_data.min()}")
print(f"  Max: {hour_data.max()}")
print(f"  Mean: {hour_data.mean()}")

# Find invalid hours
invalid_hours = np.where((hour_data < 0) | (hour_data >= 24))[0]
if len(invalid_hours) > 0:
    print(f"\n⚠️  WARNING: Found {len(invalid_hours)} invalid hour values!")
    print(f"Rows with invalid hours: {invalid_hours[:20]}")
    for idx in invalid_hours[:5]:
        print(f"  Row {idx}: {data_loader.data_s.iloc[idx]}")
else:
    print(f"\n✓ All hour values are valid in transformed data")

# Get a batch and check
print(f"\n" + "=" * 80)
print(f"Getting a batch to check hour values...")
train_next = data_loader.next_batch(
    batch_size=para.batch_size, epoch=1, is_training=True
)

try:
    x_s, day, d_o_w, hour, minute, label_s = train_next.next()
    print(f"\nBatch shapes:")
    print(f"  x_s shape: {x_s.shape}")
    print(f"  hour shape: {hour.shape}")
    print(f"  hour min: {hour.min()}, max: {hour.max()}")

    invalid_in_batch = np.where((hour < 0) | (hour >= 24))
    if len(invalid_in_batch[0]) > 0:
        print(f"\n⚠️  Invalid hour values in batch at positions: {invalid_in_batch}")
        for pos in invalid_in_batch[0][:5]:
            print(f"  Position {pos}: hour={hour[pos]}")
    else:
        print(f"✓ All hour values in batch are valid")
except Exception as e:
    print(f"Error getting batch: {e}")
    import traceback

    traceback.print_exc()
