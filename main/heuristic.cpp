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
    if (horizontal == 0.0f) return 0.0f;
    return (vertical1 + vertical2) / (2.0f * horizontal);
}

float HeuristicPipeline::CalculateMar(const std::vector<Point>& mouth) {
    if (mouth.size() < 4) return 0.0f;
    float vertical = EuclideanDistance(mouth[2], mouth[3]);
    float horizontal = EuclideanDistance(mouth[0], mouth[1]);
    if (horizontal == 0.0f) return 0.0f;
    return vertical / (2.0f * horizontal);
}

uint8_t HeuristicPipeline::GetPixelGray(const uint8_t* frame, int frame_w, int frame_h, int x, int y) {
    if (x < 0 || x >= frame_w || y < 0 || y >= frame_h) return 0;
    int idx = (y * frame_w + x) * 3;
    // Standard NTSC formula approximated with bitshifts for speed
    return (frame[idx] * 76 + frame[idx+1] * 150 + frame[idx+2] * 29) >> 8; 
}

std::vector<Point> HeuristicPipeline::ExtractEyePoints(const uint8_t* frame, int frame_w, int frame_h, int cx, int cy, int roi_w, int roi_h) {
    std::vector<Point> pts(6);
    int half_w = roi_w / 2;
    int half_h = roi_h / 2;

    // 1. Horizontal Scan for p0 (left) and p3 (right)
    // We look for the darkest points on the horizontal line passing through the eye center
    int left_bound = cx - half_w;
    int right_bound = cx + half_w;
    
    int p0_x = cx; uint8_t min_left_val = 255;
    for (int x = left_bound; x < cx; ++x) {
        uint8_t val = GetPixelGray(frame, frame_w, frame_h, x, cy);
        if (val < min_left_val) { min_left_val = val; p0_x = x; }
    }
    
    int p3_x = cx; uint8_t min_right_val = 255;
    for (int x = cx + 1; x <= right_bound; ++x) {
        uint8_t val = GetPixelGray(frame, frame_w, frame_h, x, cy);
        if (val < min_right_val) { min_right_val = val; p3_x = x; }
    }

    pts[0] = {(float)p0_x, (float)cy};
    pts[3] = {(float)p3_x, (float)cy};

    // 2. Vertical Scans at 1/3 and 2/3 distance
    int width = p3_x - p0_x;
    if (width <= 0) width = 1; // Failsafe
    int x1 = p0_x + (width / 3);
    int x2 = p0_x + (2 * width / 3);

    // Lambda helper for vertical gradient scan
    auto scan_vertical = [&](int x_scan, Point& top_pt, Point& bot_pt) {
        int best_top_y = cy; int max_neg_grad = 0; // Light -> Dark
        int best_bot_y = cy; int max_pos_grad = 0; // Dark -> Light

        for (int y = cy - half_h; y < cy + half_h; ++y) {
            int p1 = GetPixelGray(frame, frame_w, frame_h, x_scan, y);
            int p2 = GetPixelGray(frame, frame_w, frame_h, x_scan, y + 1);
            int grad = p2 - p1;

            if (y < cy && grad < max_neg_grad) { // Upper eyelid (Skin down to Eyelash)
                max_neg_grad = grad;
                best_top_y = y;
            }
            if (y >= cy && grad > max_pos_grad) { // Lower eyelid (Eye down to Skin)
                max_pos_grad = grad;
                best_bot_y = y;
            }
        }
        top_pt = {(float)x_scan, (float)best_top_y};
        bot_pt = {(float)x_scan, (float)best_bot_y};
    };

    scan_vertical(x1, pts[1], pts[5]); // Left vertical slice
    scan_vertical(x2, pts[2], pts[4]); // Right vertical slice

    return pts;
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
    uint8_t threshold = (sum_gray / count) * 0.6f; 

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
    int l_eye_cx = face.left_eye.x * frame_w;
    int l_eye_cy = face.left_eye.y * frame_h;
    int r_eye_cx = face.right_eye.x * frame_w;
    int r_eye_cy = face.right_eye.y * frame_h;
    
    Point m_left = {face.left_mouth.x * frame_w, face.left_mouth.y * frame_h};
    Point m_right = {face.right_mouth.x * frame_w, face.right_mouth.y * frame_h};

    // 2. Dynamically calculate ROI sizes based on face bounding box width
    int face_width = (face.box.x_max - face.box.x_min);
    int eye_roi_w = face_width * 0.35f;
    int eye_roi_h = face_width * 0.20f;
    int mouth_roi_h = face_width * 0.35f;

    // 3. Extract Points via Gradients
    std::vector<Point> left_eye_pts = ExtractEyePoints(frame_buffer, frame_w, frame_h, l_eye_cx, l_eye_cy, eye_roi_w, eye_roi_h);
    std::vector<Point> right_eye_pts = ExtractEyePoints(frame_buffer, frame_w, frame_h, r_eye_cx, r_eye_cy, eye_roi_w, eye_roi_h);
    std::vector<Point> mouth_pts = ExtractMouthPoints(frame_buffer, frame_w, frame_h, m_left, m_right, mouth_roi_h);

    // 4. Calculate Raw Ratios
    float l_ear = CalculateEar(left_eye_pts);
    float r_ear = CalculateEar(right_eye_pts);
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