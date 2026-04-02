# -- coding: utf-8 --
from models.inits import *


class DataClass(object):
    def __init__(self, hp=None):
        """
        :param hp:
        """
        self.hp = hp  # hyperparameter
        self.min_value = 0.000000000001
        self.input_length = self.hp.input_length  # time series length of input
        self.output_length = self.hp.output_length  # the length of prediction
        self.is_training = self.hp.is_training  # true or false
        self.divide_ratio = (
            self.hp.divide_ratio
        )  # the divide between in training set and test set ratio
        self.step = self.hp.step  # windows step
        self.site_num = self.hp.site_num
        self.file_train_s = self.hp.file_train_s
        self.normalize = self.hp.normalize  # data normalization

        self.data_s = self.get_source_data(self.file_train_s)
        self.shape_s = self.data_s.shape

        self.length = self.data_s.shape[0]  # data length
        self.max_s, self.min_s = self.get_max_min(
            self.data_s
        )  # max and min values' dictionary

        self.normalization(
            self.data_s,
            ["speed"],
            max_dict=self.max_s,
            min_dict=self.min_s,
            is_normalize=self.normalize,
        )  # normalization

    def get_source_data(self, file_path=None):
        """
        :return: Reshapes METR-LA format data for MT-STGIN model
        """
        data = pd.read_csv(file_path, encoding="utf-8")

        if "date" in data.columns:
            # Parse datetime
            data["date"] = pd.to_datetime(data["date"])

            # Extract sensor columns and features
            sensor_cols = [col for col in data.columns if col != "date"]
            num_sensors = len(sensor_cols)
            num_timestamps = len(data)

            # Pre-extract datetime features
            day_of_week_arr = data["date"].dt.dayofweek.values
            hour_arr = data["date"].dt.hour.values
            minute_arr = data["date"].dt.minute.values

            # Get speed data as numpy array (timestamps × sensors)
            speed_data = data[
                sensor_cols
            ].values  # shape: (num_timestamps, num_sensors)

            # Reshape: interleave timestamps and sensors
            output_array = np.zeros((num_timestamps * num_sensors, 6), dtype=np.float32)

            for t in range(num_timestamps):
                for s in range(num_sensors):
                    row_idx = t * num_sensors + s
                    speed_val = speed_data[t, s]
                    output_array[row_idx] = [
                        speed_val,
                        day_of_week_arr[t],
                        hour_arr[t],
                        minute_arr[t],
                        speed_val,
                        speed_val,
                    ]

            # Convert to DataFrame with proper column names
            formatted_df = pd.DataFrame(
                output_array,
                columns=[
                    "feature",
                    "day_of_week",
                    "hour",
                    "minute",
                    "pollution",
                    "speed",
                ],
            )
            return formatted_df

        return data

    def get_max_min(self, data=None):
        """
        :param data:
        :return:
        """
        min_dict = dict()
        max_dict = dict()

        for key in data.keys():
            min_dict[key] = data[key].min()
            max_dict[key] = data[key].max()
        return max_dict, min_dict

    def normalization(
        self, data, keys=None, max_dict=None, min_dict=None, is_normalize=True
    ):
        """
        :param data:
        :param keys:  is a list
        :param is_normalize:
        :return:
        """
        if is_normalize:
            for key in keys:
                data[key] = (data[key] - min_dict[key]) / (
                    max_dict[key] - min_dict[key] + self.min_value
                )

    def generator(self):
        """
        :return: yield the data of every time, input shape: [batch, site_num*(input_length+output_length)*features]
        label:   [batch, site_num, output_length]
        """
        data_s = self.data_s.values
        if self.is_training:
            low, high = 0, int(self.shape_s[0] // self.site_num * self.divide_ratio)
        else:
            low, high = int(self.shape_s[0] // self.site_num * self.divide_ratio), int(
                self.shape_s[0] // self.site_num
            )

        while low + self.input_length + self.output_length <= high:
            label = data_s[
                (low + self.input_length)
                * self.site_num : (low + self.input_length + self.output_length)
                * self.site_num,
                -1:,
            ]
            label = np.concatenate(
                [
                    label[i * self.site_num : (i + 1) * self.site_num, :]
                    for i in range(self.output_length)
                ],
                axis=1,
            )

            yield (
                data_s[
                    low * self.site_num : (low + self.input_length) * self.site_num, 5:6
                ],
                data_s[
                    low
                    * self.site_num : (low + self.input_length + self.output_length)
                    * self.site_num,
                    1,
                ],  # day_of_week
                data_s[
                    low
                    * self.site_num : (low + self.input_length + self.output_length)
                    * self.site_num,
                    1,
                ],  # day of week (same as above)
                (
                    data_s[
                        low
                        * self.site_num : (low + self.input_length + self.output_length)
                        * self.site_num,
                        2,
                    ]
                    % 24
                ).astype(
                    np.int32
                ),  # hour (0-23, ensure int32)
                (
                    data_s[
                        low
                        * self.site_num : (low + self.input_length + self.output_length)
                        * self.site_num,
                        3,
                    ]
                    // 15
                ).astype(
                    np.int32
                ),  # minute divided by 15 (0-3 for vocab_size=4)
                label,
            )
            if self.is_training:
                low += self.step
            else:
                low += self.output_length

    def get_dataset(self, batch_size, epoch, is_training=True):
        """
        Returns a tf.data.Dataset that can be iterated directly.
        Replaces the old next_batch() method that used make_one_shot_iterator().

        :param batch_size:
        :param epoch:
        :param is_training:
        :return: tf.data.Dataset
        """
        self.is_training = is_training
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=[self.input_length * self.site_num, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[(self.input_length + self.output_length) * self.site_num], dtype=tf.int32),
                tf.TensorSpec(shape=[(self.input_length + self.output_length) * self.site_num], dtype=tf.int32),
                tf.TensorSpec(shape=[(self.input_length + self.output_length) * self.site_num], dtype=tf.int32),
                tf.TensorSpec(shape=[(self.input_length + self.output_length) * self.site_num], dtype=tf.int32),
                tf.TensorSpec(shape=[self.site_num, self.output_length], dtype=tf.float32),
            ),
        )

        if self.is_training:
            dataset = dataset.shuffle(
                buffer_size=int(
                    self.shape_s[0] // self.hp.site_num * self.divide_ratio
                    - self.input_length
                    - self.output_length
                )
                // self.step
            )
            dataset = dataset.repeat(count=epoch)
        dataset = dataset.batch(batch_size=batch_size)

        return dataset

    # Keep backward compatible method name
    def next_batch(self, batch_size, epoch, is_training=True):
        """Backward compatible wrapper — returns a dataset."""
        return self.get_dataset(batch_size, epoch, is_training)
