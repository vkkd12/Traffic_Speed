import tensorflow as tf

print(f"TF version: {tf.__version__}")
print(f"GPU available: {tf.test.is_built_with_cuda()}")
print(f"GPU list: {len(tf.config.list_physical_devices('GPU'))}")
print(
    f"Running on: {'GPU' if len(tf.config.list_physical_devices('GPU')) > 0 else 'CPU'}"
)
