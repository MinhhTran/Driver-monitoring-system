#ifndef HEURISTIC_H
#define HEURISTIC_H

#include <vector>
#include <string>
#include <cstdint>

struct MTMNFace;

// Global Threshold Constants
extern const float EAR_THRESHOLD;
extern const float MAR_THRESHOLD;
extern const float PERCLOS_THRESHOLD;
extern const int YAWN_FRAME_COUNT;
extern const int WINDOW_SIZE;

// Scale-invariant spatial tracking data structure
struct Point { 
    float x; 
    float y; 
};

// Global Threshold Constants
extern const float EAR_THRESHOLD;
extern const float MAR_THRESHOLD;
extern const float PERCLOS_THRESHOLD;
extern const int YAWN_FRAME_COUNT;
extern const int WINDOW_SIZE;

// Stochastic Signal Smoothing Engine
class SimpleKalmanFilter {
public:
    float Q; // Process noise covariance
    float R; // Measurement noise covariance
    float P; // Estimation error covariance
    float X; // True state estimate
    
    SimpleKalmanFilter(float process_noise = 1e-5f, float measurement_noise = 1e-1f, 
                       float estimation_error = 1.0f, float initial_value = 0.0f);

    float update(float measurement);
};

// Encapsulated Heuristic Fatigue Pipeline to prevent global variable leakage
class HeuristicPipeline {
public:
    HeuristicPipeline();
    ~HeuristicPipeline() = default;

    // Direct interface hooks for processing
    void UpdateMetrics(const uint8_t* frame_buffer, int frame_w, int frame_h, const MTMNFace& face);
    bool IsDrowsy();
    bool IsYawning();
    float GetEAR() const { return last_smoothed_ear_; }
    float GetMAR() const { return last_smoothed_mar_; }

    // Re-exposed calculation methods for internal processing
    static float CalculateEar(float eye_height, float inter_ocular_dist);
    static float CalculateMar(const std::vector<Point>& mouth);
    static float EuclideanDistance(Point p1, Point p2);

private:
    uint8_t GetPixelGray(const uint8_t* frame, int frame_w, int frame_h, int x, int y);
    float ExtractEyeHeight(const uint8_t* frame, int frame_w, int frame_h, int cx, int cy, int roi_h);
    std::vector<Point> ExtractMouthPoints(const uint8_t* frame, int frame_w, int frame_h, Point p0, Point p1, int roi_h);

    // Internal Pipeline State Objects
    SimpleKalmanFilter ear_filter_;
    SimpleKalmanFilter mar_filter_;
    
    // Memory-Optimized Ring Buffer for PERCLOS sliding window tracking
    // Replaces standard heap-based std::vector allocation to protect SRAM
    uint8_t frame_history_[1800]; 
    int head_idx_;
    int total_elements_;
    int closed_frame_sum_;
    int yawn_counter_;
    float last_smoothed_ear_;
    float last_smoothed_mar_;
};

#endif // HEURISTIC_H