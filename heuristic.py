import math
from collections import deque

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.50
PERCLOS_THRESHOLD = 0.15  # 15% closure over the window indicates fatigue
YAWN_FRAME_COUNT = 30  # Number of consecutive frames for a yawn (approx 1 sec at 30fps)

# Sliding window
WINDOW_SIZE = 1800 
frame_buffer = deque(maxlen=WINDOW_SIZE)
yawn_counter = 0

# Landmark Indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144] 
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14] 

class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, estimation_error, initial_value=0.0):
        """
        A lightweight 1D Kalman Filter.
        """
        self.Q = process_noise      # Process noise
        self.R = measurement_noise  # Measurement noise (Higher = heavier smoothing)
        self.P = estimation_error   # Estimation error
        self.X = initial_value      # State

    def update(self, measurement):
        # Prediction Phase
        self.P = self.P + self.Q

        # Update Phase
        K = self.P / (self.P + self.R)
        self.X = self.X + K * (measurement - self.X)
        self.P = (1 - K) * self.P

        return self.X

# Initialize filters
# Layer 1: Light filter for facial landmark
landmark_filters = {}
def get_smoothed_coordinate(idx, axis, raw_val):
    key = f"{idx}_{axis}"
    if key not in landmark_filters:
        # Lower measurement noise (R) => lighter smoothing
        landmark_filters[key] = SimpleKalmanFilter(
            process_noise=1e-4, 
            measurement_noise=1e-2, 
            estimation_error=1.0, 
            initial_value=raw_val
        )
    return landmark_filters[key].update(raw_val)

# Layer 2: Heavy filter for EAR/MAR signal
ear_filter = SimpleKalmanFilter(process_noise=1e-5,
                                measurement_noise=1e-1,
                                estimation_error=1.0)
mar_filter = SimpleKalmanFilter(process_noise=1e-5,
                                measurement_noise=1e-1,
                                estimation_error=1.0)

def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def calculate_mar(landmarks, mouth_indices):
    p_left, p_right, p_top, p_bottom = [landmarks[i] for i in mouth_indices]
    vertical = euclidean_distance(p_top, p_bottom)
    horizontal = euclidean_distance(p_left, p_right)
    return vertical / horizontal

def analyze_fatigue(face_landmarks, frame_width, frame_height):
    """
    Converts normalized landmarks, applies dual-layer Kalman filtering, 
    and calculates smoothed EAR/MAR.
    """
    relevant_indices = RIGHT_EYE + LEFT_EYE + MOUTH
    
    pixel_coords = {}
    
    # Phase 1: Landmark Smoothing (Light Filter)
    # Memory Optimization: Only process the 16 specific points needed
    for idx in relevant_indices:
        landmark = face_landmarks[idx]
        raw_x = landmark.x * frame_width
        raw_y = landmark.y * frame_height
        
        smooth_x = get_smoothed_coordinate(idx, 'x', raw_x)
        smooth_y = get_smoothed_coordinate(idx, 'y', raw_y)
        
        pixel_coords[idx] = (smooth_x, smooth_y)

    # Calculate raw ratios using the smoothed landmarks
    left_ear = calculate_ear(pixel_coords, LEFT_EYE)
    right_ear = calculate_ear(pixel_coords, RIGHT_EYE)
    raw_avg_ear = (left_ear + right_ear) / 2.0
    raw_mar = calculate_mar(pixel_coords, MOUTH)

    # Phase 2: Signal Smoothing (Heavy Filter)
    smooth_ear = ear_filter.update(raw_avg_ear)
    smooth_mar = mar_filter.update(raw_mar)

    if smooth_ear < EAR_THRESHOLD:
        is_closed = 1
    else:
        is_closed = 0
    frame_buffer.append(is_closed)

    if len(frame_buffer) > 0:
        current_perclos = sum(frame_buffer) / len(frame_buffer)
    else:
        current_perclos = 0

    global yawn_counter
    is_yawning = False
    if smooth_mar > MAR_THRESHOLD:
        yawn_counter += 1
        if yawn_counter >= YAWN_FRAME_COUNT:
            is_yawning = True
    else:
        yawn_counter = 0
    
    alert_trigger = (current_perclos > PERCLOS_THRESHOLD) or is_yawning
    
    return smooth_ear, smooth_mar, current_perclos, alert_trigger
