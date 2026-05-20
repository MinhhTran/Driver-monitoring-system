#include <cmath>
#include <vector>
#include <numeric>
#include <map>
#include <string>

// Thresholds
const float EAR_THRESHOLD = 0.20f;
const float MAR_THRESHOLD = 0.50f;
const float PERCLOS_THRESHOLD = 0.15f;
const int YAWN_FRAME_COUNT = 30;
const int WINDOW_SIZE = 1800;

// Struct for Facial Landmarks
struct Point { float x; float y; };

class SimpleKalmanFilter {
public:
    float Q, R, P, X;
    
    SimpleKalmanFilter(float process_noise = 1e-5f, float measurement_noise = 1e-1f, 
                       float estimation_error = 1.0f, float initial_value = 0.0f) {
        Q = process_noise;
        R = measurement_noise;
        P = estimation_error;
        X = initial_value;
    }

    float update(float measurement) {
        // Prediction Phase
        P = P + Q;
        // Update Phase
        float K = P / (P + R);
        X = X + K * (measurement - X);
        P = (1.0f - K) * P;
        return X;
    }
};

// Global Filters & Buffers
SimpleKalmanFilter ear_filter(1e-5f, 1e-1f, 1.0f);
SimpleKalmanFilter mar_filter(1e-5f, 1e-1f, 1.0f);
std::vector<int> frame_buffer;
int yawn_counter = 0;

float euclidean_distance(Point p1, Point p2) {
    return std::hypot(p2.x - p1.x, p2.y - p1.y);
}

float calculate_ear(const std::vector<Point>& eyes) {
    float vertical1 = euclidean_distance(eyes[1], eyes[5]);
    float vertical2 = euclidean_distance(eyes[2], eyes[4]);
    float horizontal = euclidean_distance(eyes[0], eyes[3]);
    return (vertical1 + vertical2) / (2.0f * horizontal);
}

float calculate_mar(const std::vector<Point>& mouth) {
    float vertical = euclidean_distance(mouth[2], mouth[3]);
    float horizontal = euclidean_distance(mouth[0], mouth[1]);
    return vertical / (2.0f * horizontal);
}