#include "heuristic.h"
#include "deep_learning.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include "esp_log.h"

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
      l_eye_x_filter_(1e-2f, 1e-1f, 1.0f), l_eye_y_filter_(1e-2f, 1e-1f, 1.0f),
      r_eye_x_filter_(1e-2f, 1e-1f, 1.0f), r_eye_y_filter_(1e-2f, 1e-1f, 1.0f),
      m_left_x_filter_(1e-2f, 1e-1f, 1.0f), m_left_y_filter_(1e-2f, 1e-1f, 1.0f),
      m_right_x_filter_(1e-2f, 1e-1f, 1.0f), m_right_y_filter_(1e-2f, 1e-1f, 1.0f),
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

float HeuristicPipeline::CalculateEar(float eye_height, float inter_ocular_dist) {
    if (inter_ocular_dist == 0.0f) return 0.0f;
    float proxy_eye_width = inter_ocular_dist / 2.1f;
    return eye_height / proxy_eye_width;
}

float HeuristicPipeline::CalculateMar(const std::vector<Point>& mouth) {
    if (mouth.size() < 4) return 0.0f;
    float vertical = EuclideanDistance(mouth[2], mouth[3]);
    float horizontal = EuclideanDistance(mouth[0], mouth[1]);
    if (horizontal == 0.0f) return 0.0f;
    return vertical / horizontal;
}

uint8_t HeuristicPipeline::GetPixelGray(const uint8_t* frame, int frame_w, int frame_h, int x, int y) {
    if (x < 0 || x >= frame_w || y < 0 || y >= frame_h) return 0;
    int idx = (y * frame_w + x) * 3;
    // Standard NTSC formula approximated with bitshifts for speed
    return (frame[idx] * 76 + frame[idx+1] * 150 + frame[idx+2] * 29) >> 8; 
}

float HeuristicPipeline::ExtractEyeHeight(const uint8_t* frame, int frame_w, int frame_h, int cx, int cy, int roi_h) {
    int half_h = roi_h / 2;

    int top_lid_y = cy; 
    int max_neg_grad = 0; // Light -> Dark
    
    int bot_lid_y = cy; 
    int max_pos_grad = 0; // Dark -> Light

    // Scan straight up and straight down from the eye center
    for (int y = cy - half_h; y < cy + half_h; ++y) {
        int p1 = GetPixelGray(frame, frame_w, frame_h, cx, y);
        int p2 = GetPixelGray(frame, frame_w, frame_h, cx, y + 1);
        int grad = p2 - p1;

        if (y < cy && grad < max_neg_grad) { // Upper eyelid (Skin down to Eyelash)
            max_neg_grad = grad;
            top_lid_y = y;
        }
        if (y >= cy && grad > max_pos_grad) { // Lower eyelid (Eye down to Skin)
            max_pos_grad = grad;
            bot_lid_y = y;
        }
    }

    return (float)(bot_lid_y - top_lid_y);
}

std::vector<Point> HeuristicPipeline::ExtractMouthPoints(const uint8_t* frame, int frame_w, int frame_h, Point p0, Point p1, int roi_h) {
    std::vector<Point> pts(4);
    pts[0] = p0;
    pts[1] = p1;

    int cx = (int)((p0.x + p1.x) / 2);
    int cy = (int)((p0.y + p1.y) / 2);
    int half_h = roi_h / 2;

    // Calculate Adaptive Threshold: Find average brightness of the vertical column
    long sum_gray = 0;
    int count = 0;
    for (int y = cy - half_h; y <= cy + half_h; ++y) {
        sum_gray += GetPixelGray(frame, frame_w, frame_h, cx, y);
        count++;
    }
    // "Dark cavity" is defined as significantly darker than the local average
    uint8_t threshold = (sum_gray / count) * 0.85f; 

    // Scan down the center line to find top lip (first dark pixel) and bottom lip (last dark pixel)
    int top_lip_y = cy;
    int bot_lip_y = cy;
    bool found_top = false;

    for (int y = cy - half_h; y <= cy + half_h; ++y) {
        uint8_t val = GetPixelGray(frame, frame_w, frame_h, cx, y);
        if (val < threshold) {
            if (!found_top) {
                top_lip_y = y;
                found_top = true;
            }
            bot_lip_y = y; // Keeps updating until we leave the cavity
        }
    }

    pts[2] = {(float)cx, (float)top_lip_y};
    pts[3] = {(float)cx, (float)bot_lip_y};
    return pts;
}

void HeuristicPipeline::UpdateMetrics(const uint8_t* frame_buffer, int frame_w, int frame_h, const MTMNFace& face) {
    // 1. Convert normalized MTMN points to absolute pixels
    float raw_l_eye_x = face.left_eye.x * frame_w;
    float raw_l_eye_y = face.left_eye.y * frame_h;
    float raw_r_eye_x = face.right_eye.x * frame_w;
    float raw_r_eye_y = face.right_eye.y * frame_h;
    
    float raw_m_left_x = face.left_mouth.x * frame_w;
    float raw_m_left_y = face.left_mouth.y * frame_h;
    float raw_m_right_x = face.right_mouth.x * frame_w;
    float raw_m_right_y = face.right_mouth.y * frame_h;

    // Apply Kalman filters
    int l_eye_cx = (int)l_eye_x_filter_.update(raw_l_eye_x);
    int l_eye_cy = (int)l_eye_y_filter_.update(raw_l_eye_y);
    int r_eye_cx = (int)r_eye_x_filter_.update(raw_r_eye_x);
    int r_eye_cy = (int)r_eye_y_filter_.update(raw_r_eye_y);
    
    Point m_left = {face.left_mouth.x * frame_w, face.left_mouth.y * frame_h};
    Point m_right = {face.right_mouth.x * frame_w, face.right_mouth.y * frame_h};

    // 2. Dynamically calculate ROI sizes based on face bounding box width
    int face_width = (face.box.x_max - face.box.x_min);
    int eye_roi_h = face_width * 0.20f;
    int mouth_roi_h = face_width * 0.35f;

    float inter_ocular_dist = EuclideanDistance(
        {(float)l_eye_cx, (float)l_eye_cy}, 
        {(float)r_eye_cx, (float)r_eye_cy}
    );

    // 3. Extract Points via Gradients
    float l_eye_height = ExtractEyeHeight(frame_buffer, frame_w, frame_h, l_eye_cx, l_eye_cy, eye_roi_h);
    float r_eye_height = ExtractEyeHeight(frame_buffer, frame_w, frame_h, r_eye_cx, r_eye_cy, eye_roi_h);
    std::vector<Point> mouth_pts = ExtractMouthPoints(frame_buffer, frame_w, frame_h, m_left, m_right, mouth_roi_h);

    // 4. Calculate Raw Ratios
    float l_ear = CalculateEar(l_eye_height, inter_ocular_dist);
    float r_ear = CalculateEar(r_eye_height, inter_ocular_dist);
    float raw_ear = (l_ear + r_ear) / 2.0f;
    float raw_mar = CalculateMar(mouth_pts);

    // 5. Continuous Stochastic Smoothing
    last_smoothed_ear_ = ear_filter_.update(raw_ear);
    last_smoothed_mar_ = mar_filter_.update(raw_mar);

    // 6. Ring Buffer Update for PERCLOS (O(1) Execution Window)
    uint8_t current_eye_closed = (last_smoothed_ear_ < EAR_THRESHOLD) ? 1 : 0;
    if (total_elements_ == WINDOW_SIZE) {
        closed_frame_sum_ -= frame_history_[head_idx_];
    } else {
        total_elements_++;
    }
    frame_history_[head_idx_] = current_eye_closed;
    closed_frame_sum_ += current_eye_closed;
    head_idx_ = (head_idx_ + 1) % WINDOW_SIZE;

    // 7. Yawning Track Frame Logic
    if (last_smoothed_mar_ > MAR_THRESHOLD) {
        yawn_counter_++;
    } else {
        yawn_counter_ = 0; 
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