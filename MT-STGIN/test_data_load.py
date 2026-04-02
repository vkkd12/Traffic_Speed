#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple test to verify data loading works"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"CWD: {os.getcwd()}")

sys.path.insert(0, ".")

try:
    from models.hyparameter import parameter
    from models.data_next_simplified import SimplifiedDataClass
    import argparse

    print("Imports successful")

    para_obj = parameter(argparse.ArgumentParser())
    para = para_obj.get_para()
    print(f"File path: {para.file_train_s}")
    print(f"Absolute path: {os.path.abspath(para.file_train_s)}")
    print(f"Path exists: {os.path.exists(para.file_train_s)}")

    print("Creating data class...")
    data = SimplifiedDataClass(hp=para)
    print("SUCCESS - Data loaded!")
    print(f"Num timestamps: {data.num_timestamps}")
    print(f"Site num: {data.site_num}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
