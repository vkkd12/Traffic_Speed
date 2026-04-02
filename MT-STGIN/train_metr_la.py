#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training wrapper for MT-STGIN with METR-LA dataset (simplified data adapter)
Automatically runs training without requiring keyboard input
"""
import sys
import os
import traceback

# Add models to path
sys.path.insert(0, ".")

try:
    from models.hyparameter import parameter
    from models.data_next_simplified import SimplifiedDataClass
    from run_train import Model
    import argparse

    print("\n" + "=" * 80)
    print("MT-STGIN Training with METR-LA Dataset (Simplified Adapter)")
    print("=" * 80 + "\n")

    # Create hyperparameters
    para_obj = parameter(argparse.ArgumentParser())
    para = para_obj.get_para()

    print(f"Configuration:")
    print(f"  • Dataset: METR-LA (207 sensors)")
    print(f"  • Epochs: {para.epoch}")
    print(f"  • Batch size: {para.batch_size}")
    print(f"  • Learning rate: {para.learning_rate}")
    print(f"  • Input length: {para.input_length} timesteps")
    print(f"  • Output length: {para.output_length} timesteps")
    print(f"\n" + "=" * 80 + "\n")

    # Initialize and run training
    print("Creating model...")
    model = Model(para)

    print("Initializing TensorFlow session...")
    model.initialize_session()

    print("Starting training...")
    model.run_epoch()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80 + "\n")

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
