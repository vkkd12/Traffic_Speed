# -- coding: utf-8 --
"""
METR-LA Data Loader for ST-ANet Model
Adapted from MT-STGIN data loader to work with ST-ANet's input/output format

Output shapes:
- x_s: [batch * input_length, site_num, 1] - speed features
- day: [batch * (input_length + output_length), site_num] - day of week embeddings (0-6)
- hour: [batch * (input_length + output_length), site_num] - hour embeddings (0-23)
- label_s: [site_num, output_length] - prediction targets
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf


class SimplifiedDataClassST(object):
    """
    Data loader for ST-ANet with METR-LA dataset.
    Pre-loads and pre-processes METR-LA data for efficient batch generation.
    """

    def __init__(self, hp=None):
        """
        Initialize data loader for ST-ANet.

        Args:
            hp: hyperparameter object with attributes:
                - file_train_s: path to METR-LA train.csv
                - input_length: number of input timesteps
                - output_length: number of output timesteps
                - is_training: whether this is training data
                - normalize: whether to normalize data
                - site_num: number of sensors (should be 207 for METR-LA)
        """
        self.hp = hp
        self.min_value = 1e-12
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length
        self.is_training = self.hp.is_training
        self.site_num = self.hp.site_num
        self.batch_size = self.hp.batch_size

        # Load METR-LA data
        print("Loading METR-LA data...")
        self.speeds_array = self._get_source_data(self.hp.file_train_s)

        # Calculate data statistics BEFORE using them
        self.num_timestamps = self.speeds_array.shape[0]
        self.num_sensors = self.speeds_array.shape[1]
        self.possible_windows = (
            self.num_timestamps - (self.input_length + self.output_length) + 1
        )

        # Extract temporal features (needs self.num_timestamps)
        self.day_array, self.dow_array, self.hour_array, self.minute_array = (
            self._extract_temporal_features()
        )

        print(
            f"Data loaded: {self.num_timestamps} timestamps × {self.num_sensors} sensors"
        )
        print(f"Possible windows: {self.possible_windows}")
        print(
            f"Speed range: [{np.min(self.speeds_array):.2f}, {np.max(self.speeds_array):.2f}]"
        )

        # Normalization (if needed)
        self.normalize = self.hp.normalize
        if self.normalize:
            self.max_speed = np.max(self.speeds_array)
            self.min_speed = np.min(self.speeds_array)
            self.speeds_array = (self.speeds_array - self.min_speed) / (
                self.max_speed - self.min_speed + self.min_value
            )

    def _get_source_data(self, file_path):
        """
        Load METR-LA CSV and extract speed data for all sensors.

        Returns:
            speeds_array: [num_timestamps, num_sensors] numpy array
        """
        data = pd.read_csv(file_path, encoding="utf-8")

        # Remove date column if present
        if "date" in data.columns:
            sensor_cols = [col for col in data.columns if col != "date"]
        else:
            sensor_cols = list(data.columns)

        speeds_array = data[sensor_cols].values.astype(np.float32)
        return speeds_array

    def _extract_temporal_features(self):
        """
        Extract temporal features from METR-LA data.

        Returns:
            day_array: [num_timestamps] - day of month (1-31)
            dow_array: [num_timestamps] - day of week (0-6)
            hour_array: [num_timestamps] - hour (0-23)
            minute_array: [num_timestamps] - minute quantized to 0-3 (via //15)
        """
        # METR-LA data is sampled every 5 minutes starting from 2012-03-01
        # Create datetime index
        start_date = pd.Timestamp("2012-03-01")
        timestamps = pd.date_range(
            start=start_date, periods=self.num_timestamps, freq="5min"
        )

        day_array = timestamps.day.values.astype(np.int32)
        dow_array = timestamps.dayofweek.values.astype(np.int32)
        hour_array = timestamps.hour.values.astype(np.int32)
        minute_array = (timestamps.minute // 15).values.astype(np.int32)

        return day_array, dow_array, hour_array, minute_array

    def generator(self, batch_size, is_training=True):
        """
        Generate batches for training/validation.

        Yields:
            Tuple[x_s, day, hour, label_s] where:
            - x_s: [batch, input_length * site_num, 1] - input speed data
            - day: [batch, (input_length + output_length) * site_num] - day embeddings
            - hour: [batch, (input_length + output_length) * site_num] - hour embeddings
            - label_s: [batch, site_num, output_length] - output labels
        """
        if is_training:
            # Training data: use first 80% of data
            train_size = int(self.possible_windows * 0.8)
            indices = np.random.permutation(train_size)
        else:
            # Validation/test data: use last 20% of data
            train_size = int(self.possible_windows * 0.8)
            indices = np.arange(train_size, self.possible_windows)

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]

            if len(batch_indices) < batch_size:
                continue  # Skip incomplete batches

            actual_batch_size = len(batch_indices)

            # Initialize batch arrays
            x_s_batch = np.zeros(
                (actual_batch_size * self.input_length, self.site_num, 1),
                dtype=np.float32,
            )
            day_batch = np.zeros(
                (
                    actual_batch_size * (self.input_length + self.output_length),
                    self.site_num,
                ),
                dtype=np.int32,
            )
            hour_batch = np.zeros(
                (
                    actual_batch_size * (self.input_length + self.output_length),
                    self.site_num,
                ),
                dtype=np.int32,
            )
            label_batch = np.zeros(
                (actual_batch_size, self.site_num, self.output_length), dtype=np.float32
            )

            for batch_idx, window_idx in enumerate(batch_indices):
                # Extract window
                start_t = window_idx
                end_t = window_idx + self.input_length + self.output_length

                # Input features: [input_length, site_num, 1]
                input_speeds = self.speeds_array[
                    start_t : start_t + self.input_length, :
                ]
                x_s_batch[
                    batch_idx * self.input_length : (batch_idx + 1) * self.input_length,
                    :,
                    0,
                ] = input_speeds

                # Day embeddings: [input_length + output_length, site_num]
                day_vals = self.day_array[start_t:end_t]
                for t in range(len(day_vals)):
                    day_batch[
                        batch_idx * (self.input_length + self.output_length) + t, :
                    ] = day_vals[t]

                # Hour embeddings: [input_length + output_length, site_num]
                hour_vals = self.hour_array[start_t:end_t]
                for t in range(len(hour_vals)):
                    hour_batch[
                        batch_idx * (self.input_length + self.output_length) + t, :
                    ] = hour_vals[t]

                # Label: [site_num, output_length]
                output_speeds = self.speeds_array[
                    start_t + self.input_length : end_t, :
                ]
                label_batch[batch_idx, :, :] = output_speeds.T

            yield (x_s_batch, day_batch, hour_batch, label_batch)

    def next_batch(self, batch_size, epochs=1, is_training=True):
        """
        Create tf.data.Dataset from generator.

        Args:
            batch_size: batch size
            epochs: number of epochs
            is_training: whether training or validation

        Returns:
            tf.data.Dataset with proper batch structure
        """
        dataset = tf.data.Dataset.from_generator(
            lambda: self.generator(batch_size, is_training),
            output_signature=(
                tf.TensorSpec(shape=(None, self.site_num, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.site_num), dtype=tf.int32),
                tf.TensorSpec(shape=(None, self.site_num), dtype=tf.int32),
                tf.TensorSpec(
                    shape=(None, self.site_num, self.output_length), dtype=tf.float32
                ),
            ),
        )

        if is_training:
            dataset = dataset.repeat(count=epochs)

        return dataset

    @property
    def length(self):
        """Return total data length for compatibility."""
        return self.num_timestamps * self.num_sensors

    @property
    def shape_s(self):
        """Return shape of speed array."""
        return self.speeds_array.shape
