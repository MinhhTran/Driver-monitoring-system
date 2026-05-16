import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('fatigue_model_base.h5')
DATASET_DIR = 'dataset/train'

# batch_size = 1 => yield one image at a time.
calib_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    color_mode="grayscale",
    image_size=(96, 96),
    batch_size=1,
    shuffle=True
)

# The same normalization as in training
normalization_layer = tf.keras.layers.Rescaling(1./255)
calib_ds = calib_ds.map(lambda x, y: normalization_layer(x))

def representative_data_gen():
    for input_value in calib_ds.take(100):
        yield [input_value]

# Run TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Enforce full int8 quantization for both weights and activation
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open('fatigue_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantization complete")

def hex_to_c_array(hex_data, var_name):
    """Converts binary data to a C-style char array string."""
    c_str = f"const unsigned char {var_name}[] DATA_ALIGN_ATTRIBUTE = {{\n  "
    for i, val in enumerate(hex_data):
        c_str += f"0x{val:02x}, "
        if (i + 1) % 12 == 0:
            c_str += "\n  "
    c_str += "\n};\n"
    c_str += f"const int {var_name}_len = {len(hex_data)};"
    return c_str

# Read the converted tflite model
with open('fatigue_model_quantized.tflite', 'rb') as f:
    tflite_model_binary = f.read()

# Generate the C code
c_model = hex_to_c_array(tflite_model_binary, "g_fatigue_model_data")

# Save as a .cc or .h file for your ESP-IDF/Arduino project
with open('fatigue_model_data.cc', 'w') as f:
    f.write('#include "fatigue_model_data.h"\n\n')
    f.write(c_model)