import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
MODEL_PATH = 'fatigue_model_quantized.tflite'
TEST_DIR = 'dataset/test'
IMG_HEIGHT = 96
IMG_WIDTH = 96

def evaluate_quantized_model():
    print(f"Loading quantized model from: {MODEL_PATH}")
    
    # 1. Initialize the TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 2. Load the Test Dataset
    # We use batch_size=1 because the TFLite interpreter processes one image at a time
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=1,
        shuffle=False,
        color_mode="rgb"
    )
    class_names = test_ds.class_names
    print(f"Classes found: {class_names}")

    # Normalization layer (matches the 1./255 rescaling used during training)
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    y_true = []
    y_pred = []
    latencies = []

    print("Starting evaluation...")

    # 3. Iterate through the test dataset
    for images, labels in test_ds:
        # Normalize the image
        normalized_img = normalization_layer(images).numpy()

        # Apply INT8 Quantization to the input
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            input_data = (normalized_img / scale + zero_point).astype(np.int8)
        else:
            input_data = normalized_img.astype(np.float32)

        interpreter.set_tensor(input_details['index'], input_data)

        # Run Inference and measure latency
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)

        # Get the output prediction
        prediction = interpreter.get_tensor(output_details['index'])

        # Dequantize the output back to Float32
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            prediction = (prediction.astype(np.float32) - zero_point) * scale

        # Interpret the result (Assuming Sigmoid activation: > 0.5 is Class 1)
        pred_class = 1 if prediction[0][0] > 0.5 else 0

        y_true.append(labels.numpy()[0])
        y_pred.append(pred_class)

    # 4. Print Evaluation Metrics
    print("\n" + "="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Exclude the first few frames from latency calculation to account for warmup
    warmup_frames = 5
    if len(latencies) > warmup_frames:
        avg_latency = np.mean(latencies[warmup_frames:])
        print(f"\nAverage Inference Latency (excluding warmup): {avg_latency:.2f} ms")
    else:
        print(f"\nAverage Inference Latency: {np.mean(latencies):.2f} ms")

if __name__ == "__main__":
    evaluate_quantized_model()