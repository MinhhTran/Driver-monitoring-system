import cv2
import numpy as np
import tensorflow as tf

class FatigueClassifier:
    def __init__(self, model_path='fatigue_model.tflite'):
        """
        Initializes the TFLite interpreter for inference.
        This simulates the TFLite Micro environment on the ESP32-S3.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, frame, face_landmarks):
        """
        Steps for Pipeline B:
        1. Calculate bounding box from landmarks.
        2. Crop facial region.
        3. Convert to Grayscale.
        4. Resize to 96x96.
        5. Normalize.
        """
        h, w, _ = frame.shape
        
        # Get min/max coordinates for the bounding box
        coords = np.array([(int(l.x * w), int(l.y * h)) for l in face_landmarks])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        # Add some padding to the crop
        padding = 20
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
        
        # 1. Crop
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return None

        # 2. Grayscale conversion (reduces memory from 3 channels to 1)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Downsample to 96x96 (matches hardware latency solution)
        resized_roi = cv2.resize(gray_roi, (96, 96))
        
        # 4. Normalize and add batch/channel dimensions (1, 96, 96, 1)
        normalized_roi = resized_roi.astype(np.float32) / 255.0
        final_input = np.expand_dims(np.expand_dims(normalized_roi, axis=0), axis=-1)
        
        return final_input, resized_roi

    def run_inference(self, input_data):
        """
        Feeds the image into the quantized MobileNet/CNN.
        Automatically quantizes the input to INT8 and dequantizes the output.
        Returns a probability score for Fatigue state.
        """
        # 1. Quantize the input if the model expects INT8
        input_details = self.input_details[0]
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            # Apply the quantization formula and cast to int8
            input_data = (input_data / scale + zero_point).astype(np.int8)

        # Feed the data to the interpreter
        self.interpreter.set_tensor(input_details['index'], input_data)
        self.interpreter.invoke()
        
        # 2. Get the output prediction
        output_details = self.output_details[0]
        prediction = self.interpreter.get_tensor(output_details['index'])
        
        # 3. Dequantize the output back to FLOAT32 if it is INT8
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            prediction = (prediction.astype(np.float32) - zero_point) * scale

        return prediction[0][0]

def process_pipeline_b(frame, face_landmarks):
    """
    Main entry point for Deep Learning logic.
    """
    classifier = FatigueClassifier()
    processed_data = classifier.preprocess(frame, face_landmarks)
    
    if processed_data is not None:
        input_tensor, debug_crop = processed_data
        fatigue_score = classifier.run_inference(input_tensor)
        return fatigue_score, debug_crop
    
    return 0.0, None