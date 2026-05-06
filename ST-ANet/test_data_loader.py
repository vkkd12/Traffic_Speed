# -- coding: utf-8 --
"""
Test script to validate ST-ANet data loader with METR-LA dataset
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from models.hyparameter import parameter
from models.data_next_metr_la import SimplifiedDataClassST


def test_data_loader():
    """Test the ST-ANet data loader."""

    print("Testing ST-ANet Data Loader with METR-LA")
    print("=" * 60)

    # Load hyperparameters
    parser = argparse.ArgumentParser()
    para_obj = parameter(parser)
    para = para_obj.get_para()
    para.is_training = True
    para.batch_size = 128
    para.epochs = 1

    # Initialize data loader
    print("\n1. Initializing data loader...")
    try:
        data = SimplifiedDataClassST(hp=para)
        print("   ✓ Data loader initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize data loader: {e}")
        return False

    # Test batch generation
    print("\n2. Testing batch generation...")
    try:
        dataset = data.next_batch(
            batch_size=para.batch_size, epochs=1, is_training=True
        )

        for batch_num, batch in enumerate(dataset):
            if batch_num >= 3:  # Test first 3 batches
                break

            x_s, day, hour, label_s = batch

            print(f"\n   Batch {batch_num}:")
            print(f"   • x_s shape: {x_s.shape}")
            print(f"   • day shape: {day.shape}")
            print(f"   • hour shape: {hour.shape}")
            print(f"   • label_s shape: {label_s.shape}")

            # Expected shapes
            expected_x = (para.batch_size * para.input_length, para.site_num, 1)
            expected_day = (
                para.batch_size * (para.input_length + para.output_length),
                para.site_num,
            )
            expected_hour = (
                para.batch_size * (para.input_length + para.output_length),
                para.site_num,
            )
            expected_label = (para.batch_size, para.site_num, para.output_length)

            # Validate shapes
            if x_s.shape == expected_x:
                print(f"   ✓ x_s shape correct: {expected_x}")
            else:
                print(
                    f"   ✗ x_s shape mismatch! Expected {expected_x}, got {x_s.shape}"
                )
                return False

            if day.shape == expected_day:
                print(f"   ✓ day shape correct: {expected_day}")
            else:
                print(
                    f"   ✗ day shape mismatch! Expected {expected_day}, got {day.shape}"
                )
                return False

            if hour.shape == expected_hour:
                print(f"   ✓ hour shape correct: {expected_hour}")
            else:
                print(
                    f"   ✗ hour shape mismatch! Expected {expected_hour}, got {hour.shape}"
                )
                return False

            if label_s.shape == expected_label:
                print(f"   ✓ label_s shape correct: {expected_label}")
            else:
                print(
                    f"   ✗ label_s shape mismatch! Expected {expected_label}, got {label_s.shape}"
                )
                return False

            # Validate value ranges
            x_min, x_max = float(x_s.numpy().min()), float(x_s.numpy().max())
            day_min, day_max = int(day.numpy().min()), int(day.numpy().max())
            hour_min, hour_max = int(hour.numpy().min()), int(hour.numpy().max())
            label_min, label_max = float(label_s.numpy().min()), float(
                label_s.numpy().max()
            )

            print(f"   • x_s value range: [{x_min:.2f}, {x_max:.2f}]")
            print(f"   • day value range: [{day_min}, {day_max}] (expected [1-31])")
            print(f"   • hour value range: [{hour_min}, {hour_max}] (expected [0-23])")
            print(f"   • label_s value range: [{label_min:.2f}, {label_max:.2f}]")

            # Basic validation
            if day_max > 31 or day_min < 1:
                print(f"   ✗ day values out of expected range!")
                return False

            if hour_max > 23 or hour_min < 0:
                print(f"   ✗ hour values out of expected range!")
                return False

        print("\n   ✓ All batches generated successfully!")

    except Exception as e:
        print(f"   ✗ Error during batch generation: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ Data loader test PASSED!")
    return True


if __name__ == "__main__":
    success = test_data_loader()
    sys.exit(0 if success else 1)
