import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from pathlib import Path
import urllib.request

class TasksDatasetPreprocessor:
    def __init__(self, input_dir, output_dir, model_path='face_landmarker.task'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        
        # Download the task file (if hasn't)
        if not os.path.exists(self.model_path):
            print("Downloading Face Landmarker model asset for dataset preprocessor...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        # Initialize the New Tasks API Landmarker in IMAGE mode for offline speed
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Landmark Indices mapping directly for live detection
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [78, 308, 13, 14, 61, 291, 0, 17]

    def _get_bounding_box(self, landmarks, indices, w, h, padding=10):
        """Extracts a padded bounding box from NormalizedLandmarks."""
        coords = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
        
        return x_min, y_min, x_max, y_max

    def process_image(self, image_path):
        """Processes a single full-face image into a 96x96 feature patch."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            return None

        h, w, _ = frame.shape
        
        # Convert BGR to RGB and pack into MediaPipe's Image container
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run inference using the Tasks API
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None # Skip if face landmarks aren't registered

        face_landmarks = detection_result.face_landmarks[0]

        # 1. Bounding Box Isolation
        r_xmin, r_ymin, r_xmax, r_ymax = self._get_bounding_box(face_landmarks, self.RIGHT_EYE, w, h)
        l_xmin, l_ymin, l_xmax, l_ymax = self._get_bounding_box(face_landmarks, self.LEFT_EYE, w, h)
        m_xmin, m_ymin, m_xmax, m_ymax = self._get_bounding_box(face_landmarks, self.MOUTH, w, h)
        
        # 2. Extract crops
        right_eye_crop = frame[r_ymin:r_ymax, r_xmin:r_xmax]
        left_eye_crop = frame[l_ymin:l_ymax, l_xmin:l_xmax]
        mouth_crop = frame[m_ymin:m_ymax, m_xmin:m_xmax]
        
        if right_eye_crop.size == 0 or left_eye_crop.size == 0 or mouth_crop.size == 0:
            return None
            
        # 3. Structural Resizing and Stitching (Top half: 48x48 eyes, Bottom half: 96x48 mouth)
        r_eye_resized = cv2.resize(right_eye_crop, (48, 48))
        l_eye_resized = cv2.resize(left_eye_crop, (48, 48))
        top_half = np.hstack((r_eye_resized, l_eye_resized)) 
        
        mouth_resized = cv2.resize(mouth_crop, (96, 48)) 
        composite_roi = np.vstack((top_half, mouth_resized)) 
        
        # 4. Embedded Grayscale Normalization Simulation
        gray_roi = cv2.cvtColor(composite_roi, cv2.COLOR_BGR2GRAY)
        rgb_tricked_roi = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2RGB)

        return rgb_tricked_roi

    def run(self):
        """Flattens the nested NTHU directory syntax into clean training classes."""
        print(f"Scanning raw files in {self.input_dir}...")
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        processed_count = 0
        skipped_count = 0

        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    input_path = Path(root) / file
                    path_str = str(input_path).lower()
                    
                    # Sort structural branches into proper labels
                    if 'awake' in path_str:
                        target_class = 'awake'
                    elif 'fatigue' in path_str:
                        target_class = 'fatigue'
                    else:
                        continue 
                    
                    # Deduplicate overlapping image indexes by extracting the sub-class state
                    parent_folder = input_path.parent.parent.name if input_path.parent.name == 'jpgs' else input_path.parent.name
                    new_filename = f"{parent_folder}_{file}"
                    
                    output_path = self.output_dir / target_class / new_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract composite matrix
                    composite_img = self.process_image(input_path)
                    
                    if composite_img is not None:
                        cv2.imwrite(str(output_path), composite_img)
                        processed_count += 1
                    else:
                        skipped_count += 1
                        
                    if (processed_count + skipped_count) % 500 == 0:
                        print(f"Evaluated: {processed_count + skipped_count} files | Composited: {processed_count} | Skipped: {skipped_count}")

        # Clean close on underlying C++ pointers
        self.landmarker.close()
        print("\n--- Pipeline B Processing Target Complete ---")
        print(f"Total Saved Matrices: {processed_count}")
        print(f"Total Low-Confidence Dropped Matrices: {skipped_count}")

if __name__ == "__main__":
    RAW_DATASET_DIR = "nthu_cleaned/train"
    COMPOSITE_DATASET_DIR = "nthu_patched_2/train"
    
    preprocessor = TasksDatasetPreprocessor(
        input_dir=RAW_DATASET_DIR, 
        output_dir=COMPOSITE_DATASET_DIR
    )
    preprocessor.run()