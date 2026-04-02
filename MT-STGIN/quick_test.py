#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test of simplified data adapter"""
import sys

sys.path.insert(0, ".")

from models.hyparameter import parameter
from models.data_next_simplified import SimplifiedDataClass
import argparse

para_obj = parameter(argparse.ArgumentParser())
para = para_obj.get_para()

print("Testing data load...")
try:
    data = SimplifiedDataClass(hp=para)
    print(f"✓ Data loaded successfully")
    print(f"  Timestamps: {data.num_timestamps}")
    print(f"  Windows: {data.num_windows}")
    print(f"  Hour range: [{data.hour_array.min()}, {data.hour_array.max()}]")
    print(f"  Minute range: [{data.minute_array.min()}, {data.minute_array.max()}]")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    import traceback

    traceback.print_exc()
