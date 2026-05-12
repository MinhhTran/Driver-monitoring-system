import math

# Landmark Indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144] 
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14] 

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
    Converts normalized landmarks to pixel coordinates and calculates EAR/MAR.
    """
    # Convert normalized coordinates (0 to 1) to actual pixel coordinates
    pixel_coords = [(int(landmark.x * frame_width), int(landmark.y * frame_height)) for landmark in face_landmarks]
    
    # Calculate ratios
    left_ear = calculate_ear(pixel_coords, LEFT_EYE)
    right_ear = calculate_ear(pixel_coords, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2.0
    mar = calculate_mar(pixel_coords, MOUTH)
    
    return avg_ear, mar