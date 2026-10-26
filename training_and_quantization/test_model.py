import tensorflow as tf
import numpy as np
import cv2
import time
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
MODEL_PATH = 'fatigue_model_quantized(4).tflite'
EYE_TEST_DIR = 'dataset/test'
IMG_HEIGHT = 96
IMG_WIDTH = 96

# 1. Load the neutral reference image
REF_IMG_PATH = 'nthu_patched/train/awake/notdrowsy_001_glasses_nonsleepyCombination_87_notdrowsy.jpg'
ref_img = cv2.imread(REF_IMG_PATH)

# Change from dataset's RGB to OpenCV's BGR
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

# 2. Slice the bottom half (Rows 48 to 96, Columns 0 to 96)
neutral_mouth_crop = ref_img[48:96, 0:96]

# 3. Convert to a tensor and add the batch dimension (1, 48, 96, 3)
neutral_mouth_tensor = tf.convert_to_tensor(neutral_mouth_crop, dtype=tf.float32)
neutral_mouth_tensor = tf.expand_dims(neutral_mouth_tensor, axis=0)

def build_synthetic_composite(eye_img_tensor, neutral_mouth):
    """
    Transforms a single eye image tensor (B, H, W, 3) into the 
    96x96 composite format expected by the model.
    """
    # 1. Resize the single eye to the expected 48x48 patch size
    eye_48 = tf.image.resize(eye_img_tensor, [48, 48])
    
    # 2. Duplicate the eye horizontally to fill both Left and Right slots (96x48)
    # axis=2 represents the width dimension in (Batch, Height, Width, Channels)
    top_half = tf.concat([eye_48, eye_48], axis=2) 

    # 3. Stitch them vertically with the realistic neutral mouth
    # axis=1 represents the height dimension
    composite_img = tf.concat([top_half, neutral_mouth], axis=1)
    
    return composite_img

def evaluate_quantized_eye_model():
    print(f"Loading quantized model from: {MODEL_PATH}")
    
    # 1. Initialize the TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 2. Load the eye-only test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        EYE_TEST_DIR,
        image_size=(48, 48),
        batch_size=1,
        shuffle=False,
        color_mode="rgb"
    )
    class_names = test_ds.class_names
    print(f"Classes found: {class_names}")

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    y_true = []
    y_pred = []
    latencies = []

    print("Starting synthetic composite evaluation...")

    # 3. Iterate, synthesize, evaluate
    for images, labels in test_ds:
        # Build the 96x96 composite from the single eye
        composite_img = build_synthetic_composite(images, neutral_mouth_tensor)
        
        # Normalize the new composite image
        normalized_img = normalization_layer(composite_img).numpy()

        # Apply INT8 quantization to the input
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            input_data = (normalized_img / scale + zero_point).astype(np.int8)
        else:
            input_data = normalized_img.astype(np.float32)

        interpreter.set_tensor(input_details['index'], input_data)

        # Run inference
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

        # Get the output prediction
        prediction = interpreter.get_tensor(output_details['index'])

        # Dequantize
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            prediction = (prediction.astype(np.float32) - zero_point) * scale

        pred_class = 1 if prediction[0][0] > 0.5 else 0
        y_true.append(labels.numpy()[0])
        y_pred.append(pred_class)

    # 4. Print metrics
    print("\n" + "="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    warmup_frames = 5
    if len(latencies) > warmup_frames:
        avg_latency = np.mean(latencies[warmup_frames:])
        print(f"\nAverage Inference Latency (excluding warmup): {avg_latency:.2f} ms")
    else:
        print(f"\nAverage Inference Latency: {np.mean(latencies):.2f} ms")

if __name__ == "__main__":
    evaluate_quantized_eye_model()