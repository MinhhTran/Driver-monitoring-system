#ifndef DEEP_LEARNING_H
#define DEEP_LEARNING_H

#include <cstdint>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Unified struct for facial bounding boxes to prevent dynamic allocation
struct BBox {
    int x_min;
    int y_min;
    int x_max;
    int y_max;
};

// Raw 2D landing coordinate mapping from MTMN / ESP-WHO
struct LandmarkPoint {
    float x;
    float y;
};

struct MTMNFace {
    BBox box; // the entire face bounding box
    LandmarkPoint left_eye;
    LandmarkPoint right_eye;
    LandmarkPoint nose;
    LandmarkPoint left_mouth;
    LandmarkPoint right_mouth;
};

class FatigueClassifier {
public:
    FatigueClassifier();
    ~FatigueClassifier() = default;
    bool Init();
    
    // Core preprocessing: Cuts eye/mouth patches from raw camera frame and fills tensor
    bool PreprocessAndPack(const uint8_t* frame_buffer, int frame_w, int frame_h, 
                           const MTMNFace& face);

    float RunInference();

private:
    BBox GetBoxAroundCenter(LandmarkPoint center, int width, int height, 
                            int frame_w, int frame_h);

    void CropAndResizePatch(const uint8_t* src_frame, int src_w, int src_h, 
                            BBox box, uint8_t* dest_patch, int dest_w, int dest_h);

    // TFLite Micro components
    const tflite::Model* model_ = nullptr;
    tflite::MicroInterpreter* interpreter_ = nullptr;
    TfLiteTensor* input_tensor_ = nullptr;
    TfLiteTensor* output_tensor_ = nullptr;

    // Tensor Arena allocation sizing for Quantized MobileNet
    static constexpr int kTensorArenaSize = 1500 * 1024;
    uint8_t* tensor_arena_ = nullptr;
};

#endif // DEEP_LEARNING_H