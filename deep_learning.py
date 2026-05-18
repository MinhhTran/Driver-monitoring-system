import cv2
import numpy as np
import tensorflow as tf

class FatigueClassifier:
    def __init__(self, model_path='fatigue_model_quantized.tflite'):
        """
        Initializes the TFLite interpreter for inference.
        This simulates the TFLite Micro environment on the ESP32-S3.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Define MediaPipe indices based on your heuristic logic
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        # Expanded mouth indices to ensure the whole mouth is captured
        self.MOUTH = [78, 308, 13, 14, 61, 291, 0, 17] 

    def _get_bounding_box(self, landmarks, indices, w, h, padding=10):
        """Helper to get a padded bounding box for specific facial features."""
        coords = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
        
        return x_min, y_min, x_max, y_max

    def preprocess(self, frame, face_landmarks):
        """
        Feature Patch Strategy:
        1. Extract bounding boxes for Right Eye, Left Eye, and Mouth.
        2. Crop and resize: Eyes to 48x48 each, Mouth to 96x48.
        3. Stitch into a single densely packed 96x96 composite image.
        4. Convert to Grayscale -> RGB trick -> Normalize.
        """
        h, w, _ = frame.shape
        
        # 1. Get bounding boxes
        r_xmin, r_ymin, r_xmax, r_ymax = self._get_bounding_box(face_landmarks, self.RIGHT_EYE, w, h)
        l_xmin, l_ymin, l_xmax, l_ymax = self._get_bounding_box(face_landmarks, self.LEFT_EYE, w, h)
        m_xmin, m_ymin, m_xmax, m_ymax = self._get_bounding_box(face_landmarks, self.MOUTH, w, h)
        
        # 2. Crop
        right_eye_crop = frame[r_ymin:r_ymax, r_xmin:r_xmax]
        left_eye_crop = frame[l_ymin:l_ymax, l_xmin:l_xmax]
        mouth_crop = frame[m_ymin:m_ymax, m_xmin:m_xmax]
        
        # Failsafe if landmarks fall outside frame bounds
        if right_eye_crop.size == 0 or left_eye_crop.size == 0 or mouth_crop.size == 0:
            return None, None
            
        # 3. Resize and Stitch
        # Top half: Eyes side-by-side (48x48 each -> 96x48 total)
        r_eye_resized = cv2.resize(right_eye_crop, (48, 48))
        l_eye_resized = cv2.resize(left_eye_crop, (48, 48))
        top_half = np.hstack((r_eye_resized, l_eye_resized))
        
        # Bottom half: Mouth (96x48)
        mouth_resized = cv2.resize(mouth_crop, (96, 48))
        
        # Full composite (96x96)
        composite_roi = np.vstack((top_half, mouth_resized))
        
        # 4. Grayscale conversion (reduces memory from 3 channels to 1 conceptually)
        gray_roi = cv2.cvtColor(composite_roi, cv2.COLOR_BGR2GRAY)
        rgb_tricked_roi = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2RGB)
        
        # 5. Normalize and add batch/channel dimensions (1, 96, 96, 3)
        normalized_roi = rgb_tricked_roi.astype(np.float32) / 255.0
        final_input = np.expand_dims(normalized_roi, axis=0)
        
        # Return the stitched image for debugging instead of the squeezed full face
        return final_input, rgb_tricked_roi

    def run_inference(self, input_data):
        """
        Feeds the image into the quantized MobileNet/CNN.
        Automatically quantizes the input to INT8 and dequantizes the output.
        Returns a probability score for Fatigue state.
        """
        input_details = self.input_details[0]
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)
            
        self.interpreter.set_tensor(input_details['index'], input_data)
        self.interpreter.invoke()
        
        output_details = self.output_details[0]
        prediction = self.interpreter.get_tensor(output_details['index'])
        
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            prediction = (prediction.astype(np.float32) - zero_point) * scale
            
        return prediction[0][0]

#class Float32FatigueClassifier:
#    def __init__(self, model_path='fatigue_model_base.h5'):
#        self.model = tf.keras.models.load_model(model_path)

#    def run_inference(self, input_data):
#        # Directly outputs floating point probabilities
#        prediction = self.model(input_data, training=False)
#        return prediction.numpy()[0][0]
    
#def process_pipeline_b(frame, face_landmarks):
#    """
#    Main entry point for Deep Learning logic.
#    """
#    classifier = FatigueClassifier()
#    processed_data = classifier.preprocess(frame, face_landmarks)
#    
#    if processed_data is not None:
#        input_tensor, debug_crop = processed_data
#        fatigue_score = classifier.run_inference(input_tensor)
#        return fatigue_score, debug_crop
#    
#    return 0.0, None