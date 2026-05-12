import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request
import heuristic
import deep_learning

# 0. DOWNLOAD THE MODEL FILE
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading Face Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# 1. SETUP MEDIAPIPE NEW TASKS API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

dl_classifier = deep_learning.FatigueClassifier(model_path='fatigue_model.tflite')

# 2. MAIN CAMERA LOOP
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # MediaPipe requires an 'mp.Image' object in RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # VIDEO mode requires a monotonically increasing timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Phase 1: Landmarking (shared)
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            
            # Convert normalized coordinates (0 to 1) to actual pixel coordinates
            h, w, _ = frame.shape
            # Pipeline A: Heuristic
            avg_ear, mar = heuristic.analyze_fatigue(face_landmarks, w, h)
            
            # Pipeline B: DL
            dl_input, debug_crop = dl_classifier.preprocess(frame, face_landmarks)
            if dl_input is not None:
                fatigue_prob = dl_classifier.run_inference(dl_input)
                
                # Display the 96x96 grayscale input the AI model sees (Debug)
                cv2.imshow('AI Input (96x96 Gray)', debug_crop)
            else:
                fatigue_prob = 0.0

            # 3. Result
            # Heuristic
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # DL
            color = (0, 0, 255) if fatigue_prob > 0.8 else (255, 255, 0)
            cv2.putText(frame, f"DL-Fatigue: {fatigue_prob:.2%}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('DMS Prototyping - Press ESC to exit', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()