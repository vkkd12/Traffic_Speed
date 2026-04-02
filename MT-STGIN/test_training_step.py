#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the training loop step by step"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import tensorflow as tf
import numpy as np
from models.hyparameter import parameter
from models.data_next_simplified import SimplifiedDataClass
from run_train import Model
from models.utils import construct_feed_dict
import argparse

print("1. Loading config...")
para_obj = parameter(argparse.ArgumentParser())
para = para_obj.get_para()
print(f"   Batch size: {para.batch_size}, Epochs: {para.epoch}, Step: {para.step}")

print("\n2. Loading data...")
data = SimplifiedDataClass(hp=para)
print(f"   Data length: {data.length}")
print(f"   Data shape_s: {data.shape_s}")

print("\n3. Creating model...")
model = Model(para)

print("\n4. Initializing session...")
model.initialize_session()  # Initialize all variables AFTER model creation but BEFORE running any ops
model.sess.run(tf.global_variables_initializer())
print("\n5. Creating training dataset...")
train_next = data.next_batch(
    batch_size=para.batch_size, epoch=para.epoch, is_training=True
)
print(f"   Train next type: {type(train_next)}")

# Calculate number of iterations
num_iters = int(
    (
        (
            data.length // para.site_num * para.divide_ratio
            - (para.input_length + para.output_length)
        )
        // para.step
    )
    * para.epoch
    // para.batch_size
)
print(f"   Total iterations: {num_iters}")

print("\n6. Testing first batch fetch...")
try:
    print("   Calling sess.run(train_next)...")
    result = model.sess.run(train_next)
    x_s, day, d_o_w, hour, minute, label_s = result
    print(f"   ✓ Got batch!")
    print(f"     x_s shape: {x_s.shape}")
    print(f"     day shape: {day.shape}")
    print(f"     label_s shape: {label_s.shape}")

    print("\n7. Testing reshape...")
    x_s_reshaped = np.reshape(
        x_s,
        [-1, para.input_length, para.site_num, para.features],
    )
    print(f"   ✓ x_s reshaped: {x_s_reshaped.shape}")

    day_reshaped = np.reshape(day, [-1, para.site_num])
    print(f"   ✓ day reshaped: {day_reshaped.shape}")

    print("\n8. Testing feed_dict construction...")
    feed_dict = construct_feed_dict(
        x_s_reshaped,
        model.adj,
        label_s,
        day_reshaped,
        np.reshape(d_o_w, [-1, para.site_num]),
        np.reshape(hour, [-1, para.site_num]),
        np.reshape(minute, [-1, para.site_num]),
        model.placeholders,
        site_num=para.site_num,
    )
    print(f"   ✓ Feed dict created with {len(feed_dict)} keys")

    print("\n9. Testing training step...")
    feed_dict.update({model.placeholders["dropout"]: para.dropout})
    loss, _ = model.sess.run((model.loss1, model.train_op_1), feed_dict=feed_dict)
    print(f"   ✓ Training step completed! Loss: {loss:.6f}")

    print("\n✓ SUCCESS - All steps work!")

except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback

    traceback.print_exc()
