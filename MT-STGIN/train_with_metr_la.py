#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for MT-STGIN with METR-LA dataset
Uses SimplifiedDataClass for robust data loading
"""
import sys
import os
import io

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import tensorflow as tf
import numpy as np
from models.hyparameter import parameter
from models.data_next_simplified import SimplifiedDataClass
from run_train import Model
import argparse

print("\n" + "=" * 80)
print("MT-STGIN Training with METR-LA Dataset")
print("=" * 80 + "\n")

try:
    # Load configuration
    print("1. Loading configuration...")
    para_obj = parameter(argparse.ArgumentParser())
    para = para_obj.get_para()

    print(f"   • Site number: {para.site_num}")
    print(f"   • Input length: {para.input_length} timesteps")
    print(f"   • Output length: {para.output_length} timesteps")
    print(f"   • Batch size: {para.batch_size}")
    print(f"   • Epochs: {para.epoch}")
    print(f"   • Learning rate: {para.learning_rate}")

    # Load data
    print("\n2. Loading METR-LA dataset...")
    data = SimplifiedDataClass(hp=para)
    print(f"   ✓ Data loaded")
    print(f"   • Timestamps: {data.num_timestamps}")
    print(f"   • Sensors: {data.site_num}")
    print(f"   • Training windows: {data.num_windows}")
    print(
        f"   • Speed range: [{data.speeds_array.min():.2f}, {data.speeds_array.max():.2f}]"
    )

    # Create model
    print("\n3. Building model architecture...")
    model = Model(para)
    print("   ✓ Model created")

    # Initialize session
    print("\n4. Initializing TensorFlow session...")
    model.initialize_session()
    # Initialize all variables AFTER model creation
    model.sess.run(tf.global_variables_initializer())
    print("   ✓ Session initialized")

    # Train
    print("\n5. Starting training...")
    print("   " + "=" * 76)

    # Set is_training flag directly and provide stdin to avoid blocking
    model.para.is_training = True
    old_stdin = sys.stdin
    sys.stdin = io.StringIO("1\n")

    try:
        model.run_epoch()
    finally:
        sys.stdin = old_stdin

    print("   " + "=" * 76)

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80 + "\n")

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
