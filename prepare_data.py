#!/usr/bin/env python
# Simple script to split train.csv into train/val/test CSV files

import pandas as pd
import os


def split_metr_la():
    df = pd.read_csv("3S-TBLN/data/METR-LA/train.csv")
    total = len(df)
    train_idx = int(total * 0.7)
    val_idx = int(total * 0.85)

    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]

    train_df.to_csv("3S-TBLN/data/METR-LA/train_split.csv", index=False)
    val_df.to_csv("3S-TBLN/data/METR-LA/val.csv", index=False)
    test_df.to_csv("3S-TBLN/data/METR-LA/test.csv", index=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")


def split_pems_bay():
    df = pd.read_csv("3S-TBLN/data/PEMS-BAY/train.csv")
    total = len(df)
    train_idx = int(total * 0.7)
    val_idx = int(total * 0.85)

    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]

    train_df.to_csv("3S-TBLN/data/PEMS-BAY/train_split.csv", index=False)
    val_df.to_csv("3S-TBLN/data/PEMS-BAY/val.csv", index=False)
    test_df.to_csv("3S-TBLN/data/PEMS-BAY/test.csv", index=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    print("Splitting METR-LA data...")
    split_metr_la()
    print("\nSplitting PEMS-BAY data...")
    split_pems_bay()
    print("\nDone!")
