#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script to identify the exact error"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

print("=" * 80)
print("Step 1: Import TensorFlow")
print("=" * 80)
try:
    import tensorflow as tf

    print(f"✓ TensorFlow {tf.__version__} loaded")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("Step 2: Import hyperparameters")
print("=" * 80)
try:
    from models.hyparameter import parameter
    import argparse

    para_obj = parameter(argparse.ArgumentParser())
    para = para_obj.get_para()
    print(f"✓ Hyperparameters loaded")
    print(f"  File: {para.file_train_s}")
    print(f"  Site num: {para.site_num}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Step 3: Load data with SimplifiedDataClass")
print("=" * 80)
try:
    from models.data_next_simplified import SimplifiedDataClass

    print("Importing data class...")
    data = SimplifiedDataClass(hp=para)
    print(f"✓ Data loaded successfully")
    print(f"  Timestamps: {data.num_timestamps}")
    print(f"  Windows: {data.num_windows}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Step 4: Create Model")
print("=" * 80)
try:
    from run_train import Model

    print("Creating model...")
    model = Model(para)
    print(f"✓ Model created successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Step 5: Initialize session")
print("=" * 80)
try:
    print("Initializing session...")
    model.initialize_session()
    print(f"✓ Session initialized successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Step 6: Run epoch")
print("=" * 80)
try:
    print("Starting training...")
    model.run_epoch()
    print(f"✓ Training completed successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("SUCCESS - All steps completed!")
print("=" * 80)
