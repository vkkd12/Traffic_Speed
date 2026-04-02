#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check if TensorFlow is using CPU or GPU"""
import tensorflow as tf

print("\n" + "=" * 80)
print("TensorFlow Device Configuration Check")
print("=" * 80 + "\n")

print(f"TensorFlow version: {tf.__version__}")

# Check available devices
print("\nAvailable devices:")
from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()
for device in devices:
    print(f"  • {device.name} ({device.device_type})")

# Check if GPU is available
print(f"\nGPU Available: {tf.test.is_built_with_cuda()}")
print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
print(f"CPU Devices: {tf.config.list_physical_devices('CPU')}")

# Show which device TensorFlow will use by default
with tf.device("/CPU:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    c = tf.matmul(a, b)
    print(f"\nTest operation placed on: {c.device}")

print("\n" + "=" * 80)
print("Status: Currently using CPU (GPU not available or not configured)")
print("=" * 80 + "\n")
