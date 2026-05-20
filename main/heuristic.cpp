#include "heuristic.h"
#include <cmath>
#include <numeric>

// Definition of Thresholds
const float EAR_THRESHOLD = 0.20f;
const float MAR_THRESHOLD = 0.50f;
const float PERCLOS_THRESHOLD = 0.15f;
const int YAWN_FRAME_COUNT = 30;
const int WINDOW_SIZE = 1800;

SimpleKalmanFilter::SimpleKalmanFilter(float process_noise, float measurement_noise, 
                                       float estimation_error, float initial_value) 
    : Q(process_noise), R(measurement_noise), P(estimation_error), X(initial_value) {}

float SimpleKalmanFilter::update(float measurement) {
    P = P + Q;
    float K = P / (P + R);
    X = X + K * (measurement - X);
    P = (1.0f - K) * P;
    return X;
}

HeuristicPipeline::HeuristicPipeline() 
    : ear_filter_(1e-5f, 1e-1f, 1.0f), 
      mar_filter_(1e-5f, 1e-1f, 1.0f),
      head_idx_(0), 
      total_elements_(0), 
      closed_frame_sum_(0), 
      yawn_counter_(0),
      last_smoothed_ear_(0.3f),
      last_smoothed_mar_(0.1f) {
    // Zero-out the rolling buffer safely on setup
    for(int i = 0; i < WINDOW_SIZE; ++i) frame_history_[i] = 0;
}

float HeuristicPipeline::EuclideanDistance(Point p1, Point p2) {
    return std::hypot(p2.x - p1.x, p2.y - p1.y);
}

float HeuristicPipeline::CalculateEar(const std::vector<Point>& eyes) {
    if (eyes.size() < 6) return 0.0f;
    float vertical1 = EuclideanDistance(eyes[1], eyes[5]);
    float vertical2 = EuclideanDistance(eyes[2], eyes[4]);
    float horizontal = EuclideanDistance(eyes[0], eyes[3]);
    return (vertical1 + vertical2) / (2.0f * horizontal);
}

float HeuristicPipeline::CalculateMar(const std::vector<Point>& mouth) {
    if (mouth.size() < 4) return 0.0f; // Ensure safety limits
    float vertical = EuclideanDistance(mouth[2], mouth[3]);
    float horizontal = EuclideanDistance(mouth[0], mouth[1]);
    return vertical / (2.0f * horizontal);
}

void HeuristicPipeline::UpdateMetrics(const std::vector<Point>& face_landmarks) {
    // 1. Extract local target slices for calculation
    // Note: Isolate landmark coordinates parsed out of your ESP-WHO MTMN pipeline here
    float raw_ear = CalculateEar(face_landmarks); 
    float raw_mar = CalculateMar(face_landmarks);

    // 2. Continuous Stochastic Smoothing Execution Loop
    last_smoothed_ear_ = ear_filter_.update(raw_ear);
    last_smoothed_mar_ = mar_filter_.update(raw_mar);

    // 3. Ultra-Fast Ring Buffer Update for PERCLOS (O(1) Execution Window)
    uint8_t current_eye_closed = (last_smoothed_ear_ < EAR_THRESHOLD) ? 1 : 0;
    
    if (total_elements_ == WINDOW_SIZE) {
        // Subtract the expiring data item from our tracker
        closed_frame_sum_ -= frame_history_[head_idx_];
    } else {
        total_elements_++;
    }

    // Insert new item and add to sum
    frame_history_[head_idx_] = current_eye_closed;
    closed_frame_sum_ += current_eye_closed;
    
    // Cycle index wrap
    head_idx_ = (head_idx_ + 1) % WINDOW_SIZE;

    // 4. Yawning Track Frame Logic
    if (last_smoothed_mar_ > MAR_THRESHOLD) {
        yawn_counter_++;
    } else {
        yawn_counter_ = 0; // Reset if the driver closes their mouth
    }
}

bool HeuristicPipeline::IsDrowsy() {
    if (total_elements_ == 0) return false;
    float current_perclos = static_cast<float>(closed_frame_sum_) / static_cast<float>(total_elements_);
    return current_perclos >= PERCLOS_THRESHOLD;
}

bool HeuristicPipeline::IsYawning() {
    return yawn_counter_ >= YAWN_FRAME_COUNT;
}