#!/usr/bin/env python
# Reshape CSV data to match the expected format for seq2instance function

import pandas as pd
import numpy as np
from datetime import datetime
import os


def reshape_metr_la():
    """Reshape METR-LA CSV to the format expected by seq2instance"""
    df = pd.read_csv("data/METR-LA/train.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Extract numeric columns (sensor data)
    sensor_cols = [col for col in df.columns if col != "date"]
    num_sites = len(sensor_cols)
    num_timesteps = len(df)

    # Create output array: (timesteps * sites, 6)
    # columns: [index, date_str, day, hour, minute, speed_value]
    output_data = []

    for t, (idx, row) in enumerate(df.iterrows()):
        date = row["date"]
        day_of_month = date.day
        hour = date.hour
        minute = date.minute

        for s, sensor_col in enumerate(sensor_cols):
            speed = row[sensor_col]
            # Format: [index, date_str, day, hour, minute, speed]
            output_data.append(
                [
                    t,  # time index
                    f"{date.year}/{date.month}/{date.day}",  # date string
                    day_of_month,  # day
                    hour,  # hour
                    minute,  # minute
                    float(speed) if not pd.isna(speed) else 0.0,  # speed value
                ]
            )

    output_df = pd.DataFrame(
        output_data, columns=["t_idx", "date", "day", "hour", "minute", "speed"]
    )

    # Split into train/val/test
    total_rows = len(output_df)
    train_rows = int(total_rows * 0.7)
    val_rows = int(total_rows * 0.85)

    # For now, just save the full dataset - the loadData function will handle the split
    output_df.to_csv("data/METR-LA/train_reshaped.csv", index=False)

    print(f"Reshaped data shape: {output_df.shape}")
    print(f"Sample rows:\n{output_df.head()}")


def reshape_pems_bay():
    """Reshape PEMS-BAY CSV to the format expected by seq2instance"""
    df = pd.read_csv("data/PEMS-BAY/train.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Extract numeric columns (sensor data)
    sensor_cols = [col for col in df.columns if col != "date"]
    num_sites = len(sensor_cols)
    num_timesteps = len(df)

    # Create output array: (timesteps * sites, 6)
    output_data = []

    for t, (idx, row) in enumerate(df.iterrows()):
        date = row["date"]
        day_of_month = date.day
        hour = date.hour
        minute = date.minute

        for s, sensor_col in enumerate(sensor_cols):
            speed = row[sensor_col]
            output_data.append(
                [
                    t,
                    f"{date.year}/{date.month}/{date.day}",
                    day_of_month,
                    hour,
                    minute,
                    float(speed) if not pd.isna(speed) else 0.0,
                ]
            )

    output_df = pd.DataFrame(
        output_data, columns=["t_idx", "date", "day", "hour", "minute", "speed"]
    )
    output_df.to_csv("data/PEMS-BAY/train_reshaped.csv", index=False)

    print(f"Reshaped data shape: {output_df.shape}")
    print(f"Sample rows:\n{output_df.head()}")


if __name__ == "__main__":
    print("Reshaping METR-LA data...")
    reshape_metr_la()
    print("\nReshaping PEMS-BAY data...")
    reshape_pems_bay()
    print("\nDone!")
