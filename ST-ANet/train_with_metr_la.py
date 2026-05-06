# -- coding: utf-8 --
"""
ST-ANet Training Script with METR-LA Dataset
Trains ST-ANet model on METR-LA traffic speed data (207 sensors)
"""

import argparse
import os
import sys
import io
import numpy as np
import tensorflow as tf

# Import from local models
sys.path.insert(0, os.path.dirname(__file__))
from models.hyparameter import parameter
from models.data_next_metr_la import SimplifiedDataClassST
from run import Model


def main():
    """Main training function."""

    print("=" * 80)
    print("ST-ANet Training with METR-LA Dataset")
    print("=" * 80)

    # 1. Load hyperparameters
    print("\n1. Loading configuration...")
    parser = argparse.ArgumentParser()
    para_obj = parameter(parser)
    para = para_obj.get_para()

    print(f"   • Site number: {para.site_num}")
    print(f"   • Input length: {para.input_length} timesteps")
    print(f"   • Output length: {para.output_length} timesteps")
    print(f"   • Batch size: {para.batch_size}")
    print(f"   • Epochs: {para.epochs}")
    print(f"   • Learning rate: {para.learning_rate}")

    # 2. Load METR-LA dataset
    print("\n2. Loading METR-LA dataset...")
    para.is_training = True
    data = SimplifiedDataClassST(hp=para)
    print(f"   [OK] Data loaded")
    print(f"   • Timestamps: {data.num_timestamps}")
    print(f"   • Sensors: {data.num_sensors}")
    print(f"   • Training windows: {data.possible_windows}")
    print(
        f"   • Speed range: [{np.min(data.speeds_array):.2f}, {np.max(data.speeds_array):.2f}]"
    )

    # 3. Build model
    print("\n3. Building model architecture...")
    model = Model(para)
    print(f"   [OK] Model created")

    # 4. Training loop
    print("\n4. Starting training...")
    print("   " + "=" * 76)

    train_dataset = data.next_batch(
        batch_size=para.batch_size, epochs=para.epochs, is_training=True
    )

    step = 0
    epoch = 0
    steps_per_epoch = data.possible_windows // para.batch_size

    for batch in train_dataset:
        # Unpack batch
        x_s, day, hour, label_s = batch

        # Reshape for model input
        # x_s: [batch * input_len, site_num, 1] -> keep as is (already correct shape)
        # day: [batch * (input + output), site_num]
        # hour: [batch * (input + output), site_num]
        # label: [batch, site_num, output_len]

        actual_batch_size = x_s.shape[0] // para.input_length

        if actual_batch_size != para.batch_size:
            continue

        # Training step
        try:
            loss = model.train_step(x_s, day, hour, label_s)
            step += 1

            # Print progress
            current_epoch = step // steps_per_epoch
            step_in_epoch = step % steps_per_epoch

            if step % 10 == 0:
                print(
                    f"Epoch {current_epoch + 1}/{para.epochs} | Step {step_in_epoch}/{steps_per_epoch} | Loss: {loss.numpy():.6f}"
                )

        except Exception as e:
            print(f"   ✗ Error during training step {step}: {str(e)}")
            import traceback

            traceback.print_exc()
            break

        # Checkpoint every 100 steps
        if step % 100 == 0:
            if not os.path.exists(para.save_path):
                os.makedirs(para.save_path)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(
                file_prefix=os.path.join(para.save_path, f"model_step{step}")
            )
            print(f"   [OK] Checkpoint saved at step {step}")

    print("   " + "=" * 76)
    print(f"\n5. Training Complete!")
    print(f"   • Total steps: {step}")
    print(f"   • Total epochs: {step // steps_per_epoch}")

    # Save final model
    if not os.path.exists(para.save_path):
        os.makedirs(para.save_path)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(file_prefix=os.path.join(para.save_path, "model_final"))
    print(f"   [OK] Final model saved to {para.save_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
