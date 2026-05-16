import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import tracemalloc
import csv
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

dl_classifier = deep_learning.FatigueClassifier(model_path='fatigue_model_quantized(2).tflite')
def monitor_performance(func, *args):
    """Wrapper to measure execution time and peak memory."""
    tracemalloc.start()
    start_time = time.perf_counter()
    
    result = func(*args)
    
    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latency_ms = (end_time - start_time) * 1000
    peak_kb = peak / 1024
    return result, latency_ms, peak_kb

# CSV data logging setup
csv_file = open('dms_performance_log(2).csv', mode='w', newline='')
log_writer = csv.writer(csv_file)
# Headers mapped to your methodology's evaluation criteria
log_writer.writerow([
    'Timestamp', 'Condition', 'Pipeline', 'Latency_ms', 
    'Peak_RAM_KB', 'EAR', 'MAR', 'PERCLOS', 'DL_Prob', 
    'System_Alert', 'Ground_Truth_Fatigue'
])

# Testing State Variables
current_condition = "Baseline"
ground_truth_fatigue = False
logging_interval = 0.05 # 0.5 means 2 logs per second)
last_log_time = 0

# 2. MAIN CAMERA LOOP
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('1'): current_condition = "Baseline"
        elif key == ord('2'): current_condition = "Low-Light"
        elif key == ord('3'): current_condition = "Rapid Transition"
        elif key == ord('4'): current_condition = "Occlusion"
        elif key == ord('s') or key == ord('S'): 
            # Toggle ground truth for accuracy matrix
            ground_truth_fatigue = not ground_truth_fatigue

        # MediaPipe requires an mp.Image object in RGB format
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
            (heuristic_result), lat_h, mem_h = monitor_performance(
                heuristic.analyze_fatigue, face_landmarks, w, h
            )
            avg_ear, mar, perclos, alert = heuristic_result

            # Pipeline B: DL
            def dl_wrapper():
                inp, dbg = dl_classifier.preprocess(frame, face_landmarks)
                if inp is not None:
                    return dl_classifier.run_inference(inp), dbg
                return 0.0, None
            
            (dl_result), lat_dl, mem_dl = monitor_performance(dl_wrapper)
            fatigue_prob, debug_crop = dl_result

            if debug_crop is not None:
                cv2.imshow('AI Input (96x96 Gray)', debug_crop)

            # 3. Result
            current_time = time.strftime('%H:%M:%S')
            current_time_sec = time.time()
            
            if (current_time_sec - last_log_time) >= logging_interval:
                print(f"\n[{current_time}] Condition: {current_condition} | GT Fatigue: {ground_truth_fatigue}")
                print(f"  HEURISTIC -> Latency: {lat_h:.2f}ms | Peak RAM: {mem_h:.2f}KB | Alert: {alert}")
                print(f"  DEEP LRN  -> Latency: {lat_dl:.2f}ms | Peak RAM: {mem_dl:.2f}KB | Prob: {fatigue_prob:.2%}")

                log_writer.writerow([
                    current_time, current_condition, 'Heuristic', round(lat_h, 2), 
                    round(mem_h, 2), round(avg_ear, 3), round(mar, 3), round(perclos, 3), 
                    '', alert, ground_truth_fatigue
                ])
                log_writer.writerow([
                    current_time, current_condition, 'Deep_Learning', round(lat_dl, 2), 
                    round(mem_dl, 2), '', '', '', round(fatigue_prob, 3), 
                    fatigue_prob > 0.8, ground_truth_fatigue
                ])
                last_log_time = current_time_sec

            # Condition and ground truth
            cv2.putText(frame, f"Cond: {current_condition}", (360, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            gt_text = "GT: FATIGUE" if ground_truth_fatigue else "GT: AWAKE"
            gt_color = (0, 0, 255) if ground_truth_fatigue else (0, 255, 0)
            cv2.putText(frame, gt_text, (360, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)

            # Heuristic
            status_text = "FATIGUE DETECTED!" if alert else "NORMAL"
            status_color = (0, 0, 255) if alert else (0, 255, 0)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.2%}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

            # DL
            color = (0, 0, 255) if fatigue_prob > 0.8 else (255, 255, 0)
            cv2.putText(frame, f"DL-Fatigue: {fatigue_prob:.2%}", (360, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('DMS Prototyping - Press ESC to exit', frame)

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
