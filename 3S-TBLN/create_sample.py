#!/usr/bin/env python
# Create a small sample dataset for testing

import pandas as pd
import numpy as np


def create_sample_metr_la():
    """Create a small sample dataset from METR-LA for testing"""
    df = pd.read_csv("data/METR-LA/train_reshaped.csv")

    # Take every 100th row to create a smaller dataset
    sample_df = df.iloc[::100, :].reset_index(drop=True)

    sample_df.to_csv("data/METR-LA/train_sample.csv", index=False)
    print(f"Created sample with {len(sample_df)} rows")
    print(sample_df.head())


if __name__ == "__main__":
    create_sample_metr_la()
