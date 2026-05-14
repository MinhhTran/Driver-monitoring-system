import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('fatigue_model_base.h5')

def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 96, 96, 1).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open('fatigue_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantization complete")