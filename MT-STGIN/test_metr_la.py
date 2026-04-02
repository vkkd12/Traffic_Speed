#!/usr/bin/env python
# -- coding: utf-8 --
"""
Test script to verify METR-LA dataset loading
"""
from models.hyparameter import parameter
from models.data_next import DataClass
import argparse

# Create parameter object
para = parameter(argparse.ArgumentParser())
hp = para.get_para()

print("=" * 80)
print("METR-LA Configuration Test")
print("=" * 80)
print(f"Site number: {hp.site_num}")
print(f"Training file: {hp.file_train_s}")
print(f"Adjacency file: {hp.file_adj}")
print(f"Input length: {hp.input_length}")
print(f"Output length: {hp.output_length}")

print("\n--- Loading Data ---")
try:
    data = DataClass(hp=hp)
    print(f"✓ Data loaded successfully!")
    print(f"Data shape: {data.shape_s}")
    print(f"Data length: {data.length}")
    print(f"Max values: {list(data.max_s.keys())[:5]}")
    print(f"Min values: {list(data.min_s.keys())[:5]}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
