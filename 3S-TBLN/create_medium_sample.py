#!/usr/bin/env python
# Create a larger sample dataset for testing

import pandas as pd


def create_sample_metr_la():
    """Create a medium sample dataset from METR-LA for testing"""
    df = pd.read_csv("data/METR-LA/train_reshaped.csv")

    # Take every 10th row to create a medium dataset
    sample_df = df.iloc[::10, :].reset_index(drop=True)

    sample_df.to_csv("data/METR-LA/train_medium.csv", index=False)
    print(f"Created medium sample with {len(sample_df)} rows")
    print(
        f"Sample timesteps: {len(sample_df) // 207}"
    )  # 207 is the number of METR-LA sites
    print(sample_df.head())


if __name__ == "__main__":
    create_sample_metr_la()
