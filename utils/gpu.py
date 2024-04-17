import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Check if GPU device is available
if tf.test.is_gpu_available():
    print("GPU device is available")
else:
    print("GPU device is not available")