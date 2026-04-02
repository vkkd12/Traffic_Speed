#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal training launcher"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Log all output
log_file = open("training.log", "w")
sys.stdout = log_file
sys.stderr = log_file

try:
    print("=" * 80)
    print("Starting MT-STGIN training with METR-LA dataset")
    print("=" * 80)

    sys.path.insert(0, ".")
    from models.hyparameter import parameter
    from models.data_next_simplified import SimplifiedDataClass
    from run_train import Model
    import argparse

    print("\n1. Loading hyperparameters...")
    para_obj = parameter(argparse.ArgumentParser())
    para = para_obj.get_para()
    print(f"   Dataset: {para.file_train_s}")
    print(f"   Batch size: {para.batch_size}")
    print(f"   Epochs: {para.epoch}")

    print("\n2. Creating model...")
    model = Model(para)

    print("\n3. Initializing session...")
    model.initialize_session()

    print("\n4. Starting training...")
    model.run_epoch()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

finally:
    log_file.close()
