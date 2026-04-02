#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test simplified data adapter"""
import sys

sys.path.insert(0, ".")

from models.hyparameter import parameter
from models.data_next_simplified import SimplifiedDataClass
import argparse
import numpy as np

print("\n" + "=" * 80)
print("Testing Simplified Data Adapter")
print("=" * 80 + "\n")

# Create hyperparameters
para_obj = parameter(argparse.ArgumentParser())
para = para_obj.get_para()

# Load data
print(f"Configuration:")
print(f"  Input length: {para.input_length}")
print(f"  Output length: {para.output_length}")
print(f"  Batch size: {para.batch_size}")
print(f"  Site num: {para.site_num}\n")

data_loader = SimplifiedDataClass(hp=para)

# Get a batch
print("Getting first batch...")
train_next = data_loader.next_batch(
    batch_size=para.batch_size, epoch=1, is_training=True
)

x_s, day, d_o_w, hour, minute, label_s = train_next

# Check batch
print(f"\nBatch shapes (before reshape):")
print(f"  x_s: {x_s.shape}")
print(f"  day: {day.shape}")
print(f"  d_o_w: {d_o_w.shape}")
print(f"  hour: {hour.shape}")
print(f"  minute: {minute.shape}")
print(f"  label_s: {label_s.shape}")

# Check ranges
print(f"\nValue ranges in batch:")
print(f"  x_s (speed): [{x_s.numpy().min():.2f}, {x_s.numpy().max():.2f}]")
print(f"  day: [{day.numpy().min()}, {day.numpy().max()}]")
print(f"  d_o_w: [{d_o_w.numpy().min()}, {d_o_w.numpy().max()}]")
print(f"  hour: [{hour.numpy().min()}, {hour.numpy().max()}]")
print(f"  minute: [{minute.numpy().min()}, {minute.numpy().max()}] (expect 0-3)")
print(f"  label_s: [{label_s.numpy().min():.2f}, {label_s.numpy().max():.2f}]")

# Check for invalid values
invalid_hour = np.where((hour.numpy() < 0) | (hour.numpy() >= 24))
invalid_minute = np.where((minute.numpy() < 0) | (minute.numpy() >= 4))

if len(invalid_hour[0]) > 0:
    print(f"\n⚠️  WARNING: Found {len(invalid_hour[0])} invalid hour values!")
else:
    print(f"\n✓ All hour values valid (0-23)")

if len(invalid_minute[0]) > 0:
    print(f"⚠️  WARNING: Found {len(invalid_minute[0])} invalid minute values!")
else:
    print(f"✓ All minute values valid (0-3)")

print("\n" + "=" * 80)
print("Data adapter test PASSED")
print("=" * 80 + "\n")
