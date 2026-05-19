import os
import cv2
import math
import shutil
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = 'face_landmarker.task'
SOURCE_DIR = 'nthu/train'
OUTPUT_DIR = 'nthu_cleaned/train'

# Strict mathematical thresholds based on your heuristic logic
EAR_THRESHOLD = 0.21  # Frames <= 0.21 are forced to Fatigue (Closed)
MAR_THRESHOLD = 0.50

# Exact landmark profiles from your Heuristic pipeline
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 308, 13, 14]

# --- HELPER GEOMETRY FUNCTIONS ---
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_ear(landmarks, eye_indices, w, h):
    """Maps normalized landmarks to pixel bounds and computes scaling-invariant EAR."""
    coords = [(l.x * w, l.y * h) for l in landmarks]
    p1, p2, p3, p4, p5, p6 = [coords[i] for i in eye_indices]
    
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def calculate_mar(landmarks, mouth_indices, w, h):
    """Computes scaling-invariant MAR using 4-point extraction."""
    coords = [(l.x * w, l.y * h) for l in landmarks]
    p_left, p_right, p_top, p_bottom = [coords[i] for i in mouth_indices]
    
    vertical = euclidean_distance(p_top, p_bottom)
    horizontal = euclidean_distance(p_left, p_right)
    
    if horizontal == 0: 
        return 0
    return vertical / (2.0 * horizontal)

# --- INIT MEDIAPIPE IMAGE PIPELINE ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Optimized for fast static image arrays
    num_faces=1
)

# Target Subdirectories
os.makedirs(os.path.join(OUTPUT_DIR, 'awake'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'fatigue'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'rejected'), exist_ok=True)

print("Starting EAR feature filtering process...")

counter_awake = 0
counter_fatigue = 0
counter_rejected = 0

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    # Walk through both 'awake' and 'fatigue' polluted source subdirectories
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(root, file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            h, w, _ = frame.shape
            
            # Convert channel order for MediaPipe vision tasks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Run landmark tracking frame execution
            detection_result = landmarker.detect(mp_image)
            
            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
                
                # Extract mathematical EAR baseline for this specific image file
                left_ear = calculate_ear(face_landmarks, LEFT_EYE_INDICES, w, h)
                right_ear = calculate_ear(face_landmarks, RIGHT_EYE_INDICES, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(face_landmarks, MOUTH_INDICES, w, h)
                
                # Binary Re-routing based on physical eye tracking state
                if avg_ear <= EAR_THRESHOLD or mar > MAR_THRESHOLD:
                    dest_path = os.path.join(OUTPUT_DIR, 'fatigue', file)
                    shutil.copy(img_path, dest_path)
                    counter_fatigue += 1
                else:
                    dest_path = os.path.join(OUTPUT_DIR, 'awake', file)
                    shutil.copy(img_path, dest_path)
                    counter_awake += 1
            else:
                # If an adverse lighting artifact breaks landmarking, isolate it
                dest_path = os.path.join(OUTPUT_DIR, 'rejected', file)
                shutil.copy(img_path, dest_path)
                counter_rejected += 1

print("\n" + "="*40)
print("DATASET FILTERING COMPLETE")
print("="*40)
print(f"True Awake Frames (Eyes Open):   {counter_awake}")
print(f"True Fatigue Frames (Eyes Closed): {counter_fatigue}")
print(f"Rejected Frames (No Landmarks):  {counter_rejected}")
print(f"Cleaned output directory saved at: {OUTPUT_DIR}")