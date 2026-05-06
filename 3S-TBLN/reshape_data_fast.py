#!/usr/bin/env python
# Fast vectorized reshape of CSV data

import pandas as pd
import numpy as np


def reshape_metr_la_fast():
    """Reshape METR-LA CSV using vectorized operations"""
    print("Loading data...")
    df = pd.read_csv("data/METR-LA/train.csv")
    df["date"] = pd.to_datetime(df["date"])

    sensor_cols = [col for col in df.columns if col != "date"]
    num_sites = len(sensor_cols)
    num_timesteps = len(df)

    print(f"Data shape: {num_timesteps} timesteps x {num_sites} sites")

    # Extract date components
    dates_str = df["date"].dt.strftime("%Y/%m/%d").values
    days = df["date"].dt.day.values
    hours = df["date"].dt.hour.values
    minutes = df["date"].dt.minute.values

    # Create time index for each timestep
    t_indices = np.arange(num_timesteps)

    # Get sensor data
    sensor_data = df[sensor_cols].values

    # Reshape: for each timestep, create num_sites rows
    output_t = np.repeat(t_indices, num_sites)
    output_dates = np.repeat(dates_str, num_sites)
    output_days = np.repeat(days, num_sites)
    output_hours = np.repeat(hours, num_sites)
    output_minutes = np.repeat(minutes, num_sites)
    output_speed = sensor_data.flatten()

    output_df = pd.DataFrame(
        {
            "t_idx": output_t,
            "date": output_dates,
            "day": output_days,
            "hour": output_hours,
            "minute": output_minutes,
            "speed": output_speed,
        }
    )

    print(f"Reshaped to: {output_df.shape}")
    output_df.to_csv("data/METR-LA/train_reshaped.csv", index=False)
    print("Saved to data/METR-LA/train_reshaped.csv")


def reshape_pems_bay_fast():
    """Reshape PEMS-BAY CSV using vectorized operations"""
    print("Loading data...")
    df = pd.read_csv("data/PEMS-BAY/train.csv")
    df["date"] = pd.to_datetime(df["date"])

    sensor_cols = [col for col in df.columns if col != "date"]
    num_sites = len(sensor_cols)
    num_timesteps = len(df)

    print(f"Data shape: {num_timesteps} timesteps x {num_sites} sites")

    # Extract date components
    dates_str = df["date"].dt.strftime("%Y/%m/%d").values
    days = df["date"].dt.day.values
    hours = df["date"].dt.hour.values
    minutes = df["date"].dt.minute.values

    # Create time index
    t_indices = np.arange(num_timesteps)

    # Get sensor data
    sensor_data = df[sensor_cols].values

    # Reshape
    output_t = np.repeat(t_indices, num_sites)
    output_dates = np.repeat(dates_str, num_sites)
    output_days = np.repeat(days, num_sites)
    output_hours = np.repeat(hours, num_sites)
    output_minutes = np.repeat(minutes, num_sites)
    output_speed = sensor_data.flatten()

    output_df = pd.DataFrame(
        {
            "t_idx": output_t,
            "date": output_dates,
            "day": output_days,
            "hour": output_hours,
            "minute": output_minutes,
            "speed": output_speed,
        }
    )

    print(f"Reshaped to: {output_df.shape}")
    output_df.to_csv("data/PEMS-BAY/train_reshaped.csv", index=False)
    print("Saved to data/PEMS-BAY/train_reshaped.csv")


if __name__ == "__main__":
    print("=== Reshaping METR-LA ===")
    reshape_metr_la_fast()
    print("\n=== Reshaping PEMS-BAY ===")
    reshape_pems_bay_fast()
    print("\nDone!")
