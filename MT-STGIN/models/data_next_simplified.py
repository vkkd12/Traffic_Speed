# -- coding: utf-8 --
"""
Simplified data adapter for METR-LA dataset
Outputs data in exact format expected by MT-STGIN model
"""
from models.inits import *


class SimplifiedDataClass(object):
    """Simplified data loader that pre-processes all features correctly."""

    def __init__(self, hp=None):
        """Initialize data loader."""
        self.hp = hp
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length
        self.divide_ratio = self.hp.divide_ratio
        self.step = self.hp.step
        self.site_num = self.hp.site_num
        self.normalize = self.hp.normalize
        self.min_value = 0.000000000001

        # Load and pre-process data
        self._load_data()

    def _load_data(self):
        """Load METR-LA data and pre-process all features."""
        print("Loading METR-LA data...")

        # Resolve file path - handle relative paths properly
        file_path = self.hp.file_train_s
        if not os.path.isabs(file_path):
            # Convert relative path to absolute using current working directory
            file_path = os.path.abspath(file_path)

        print(f"Data file path: {file_path}")
        data = pd.read_csv(file_path, encoding="utf-8")

        # Extract date and sensor columns
        data["date"] = pd.to_datetime(data["date"])
        sensor_cols = [col for col in data.columns if col != "date"]

        # Extract and pre-process temporal features
        day_values = data["date"].dt.day.values.astype(np.int32)  # 1-31
        dow_values = data["date"].dt.dayofweek.values.astype(np.int32)  # 0-6
        hour_values = data["date"].dt.hour.values.astype(np.int32)  # 0-23
        minute_values = (data["date"].dt.minute.values // 15).astype(np.int32)  # 0-3

        # Get speed data
        speeds = data[sensor_cols].values.astype(
            np.float32
        )  # (num_timestamps, num_sensors)

        # Number of timestamps
        self.num_timestamps = len(data)

        # Pre-create all temporal features arrays
        # These will be indexed as: times[timestamp_idx] to get the feature for that timestamp
        self.day_array = day_values
        self.dow_array = dow_values
        self.hour_array = hour_values
        self.minute_array = minute_values
        self.speeds_array = speeds

        # Calculate lengths
        self.num_windows = self.num_timestamps - (
            self.input_length + self.output_length
        )

        print(
            f"Data loaded: {self.num_timestamps} timestamps × {self.site_num} sensors"
        )
        print(f"Possible windows: {self.num_windows}")
        print(f"Speed range: [{speeds.min():.2f}, {speeds.max():.2f}]")

        # Get max/min for normalization
        self.max_s = {"speed": float(speeds.max())}
        self.min_s = {"speed": float(speeds.min())}

        # For compatibility with training loop
        self.length = self.num_timestamps * self.site_num
        self.shape_s = self.speeds_array.shape

    def generator(self):
        """Simple generator yielding properly formatted batches."""
        is_training = True
        step_size = self.step if is_training else self.output_length

        # Calculate split point (80% train, 20% test)
        split_point = int(self.num_timestamps * self.divide_ratio)

        if is_training:
            t_start = 0
            t_end = split_point
        else:
            t_start = split_point
            t_end = self.num_timestamps

        # Generate windows
        t = t_start
        while t + self.input_length + self.output_length <= t_end:
            # Extract input features (input_length timestamps × site_num sensors)
            input_speeds = []  # (input_length * site_num,)
            all_day = []  # (input_length * site_num + output_length * site_num,)
            all_dow = []
            all_hour = []
            all_minute = []

            # Input window: collect features for input_length timestamps
            for time_idx in range(t, t + self.input_length):
                for sensor_idx in range(self.site_num):
                    input_speeds.append(self.speeds_array[time_idx, sensor_idx])
                    all_day.append(self.day_array[time_idx])
                    all_dow.append(self.dow_array[time_idx])
                    all_hour.append(self.hour_array[time_idx])
                    all_minute.append(self.minute_array[time_idx])

            # Output window: collect features for output_length timestamps
            output_speeds = []  # (output_length * site_num,)
            for time_idx in range(
                t + self.input_length, t + self.input_length + self.output_length
            ):
                for sensor_idx in range(self.site_num):
                    output_speeds.append(self.speeds_array[time_idx, sensor_idx])
                    all_day.append(self.day_array[time_idx])
                    all_dow.append(self.dow_array[time_idx])
                    all_hour.append(self.hour_array[time_idx])
                    all_minute.append(self.minute_array[time_idx])

            # Create output arrays with correct dtypes
            x_s = np.array(input_speeds, dtype=np.float32).reshape(-1, 1)
            label_s = np.array(output_speeds, dtype=np.float32).reshape(
                self.site_num, self.output_length
            )

            day_out = np.array(all_day, dtype=np.int32)
            dow_out = np.array(all_dow, dtype=np.int32)
            hour_out = np.array(all_hour, dtype=np.int32)
            minute_out = np.array(all_minute, dtype=np.int32)

            yield (x_s, day_out, dow_out, hour_out, minute_out, label_s)

            t += step_size

    def next_batch(self, batch_size, epoch, is_training=True):
        """Create batched dataset from generator."""
        self.is_training = is_training

        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(
                tf.float32,
                tf.int32,
                tf.int32,
                tf.int32,
                tf.int32,
                tf.float32,
            ),
            output_shapes=(
                tf.TensorShape([self.input_length * self.site_num, 1]),
                tf.TensorShape(
                    [
                        self.input_length * self.site_num
                        + self.output_length * self.site_num
                    ]
                ),
                tf.TensorShape(
                    [
                        self.input_length * self.site_num
                        + self.output_length * self.site_num
                    ]
                ),
                tf.TensorShape(
                    [
                        self.input_length * self.site_num
                        + self.output_length * self.site_num
                    ]
                ),
                tf.TensorShape(
                    [
                        self.input_length * self.site_num
                        + self.output_length * self.site_num
                    ]
                ),
                tf.TensorShape([self.site_num, self.output_length]),
            ),
        )

        if is_training:
            dataset = dataset.repeat(count=epoch)

        dataset = dataset.batch(batch_size=batch_size)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()
