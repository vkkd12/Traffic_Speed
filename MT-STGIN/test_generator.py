#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test SimplifiedDataClass generator output"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from models.hyparameter import parameter
from models.data_next_simplified import SimplifiedDataClass
import argparse

para_obj = parameter(argparse.ArgumentParser())
para = para_obj.get_para()

print("1. Loading data...")
data = SimplifiedDataClass(hp=para)
print(f"   Data loaded: {data.num_timestamps} timestamps, {data.num_windows} windows")

print("\n2. Testing generator - first 3 batches:")
gen = data.generator()
for i in range(3):
    try:
        batch = next(gen)
        x_s, day_out, dow_out, hour_out, minute_out, label_s = batch
        print(f"\n   Batch {i}:")
        print(
            f"     x_s shape: {x_s.shape}, dtype: {x_s.dtype}, min/max: {x_s.min():.2f}/{x_s.max():.2f}"
        )
        print(
            f"     day shape: {day_out.shape}, range: [{day_out.min()}, {day_out.max()}]"
        )
        print(
            f"     dow shape: {dow_out.shape}, range: [{dow_out.min()}, {dow_out.max()}]"
        )
        print(
            f"     hour shape: {hour_out.shape}, range: [{hour_out.min()}, {hour_out.max()}]"
        )
        print(
            f"     minute shape: {minute_out.shape}, range: [{minute_out.min()}, {minute_out.max()}]"
        )
        print(f"     label_s shape: {label_s.shape}, dtype: {label_s.dtype}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        break

print("\n3. Testing dataset batching...")
try:
    dataset = data.next_batch(batch_size=128, epoch=0, is_training=True)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    print("   Dataset created successfully")
    print(f"   Next element type: {type(next_element)}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback

    traceback.print_exc()
