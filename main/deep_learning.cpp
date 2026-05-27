#include "deep_learning.h"
#include "fatigue_model_data.h"
#include <algorithm>
#include <cmath>

// Standard Landmark Indices used by your preprocessing script
const int RIGHT_EYE_INDICES[] = {33, 160, 158, 133, 153, 144};
const int LEFT_EYE_INDICES[]  = {362, 385, 387, 263, 373, 380};
const int MOUTH_INDICES[]     = {78, 308, 13, 14, 61, 291, 0, 17};

FatigueClassifier::FatigueClassifier() {}

bool FatigueClassifier::Init() {
    // Map the compiled model array into the TFLite runtime space
    model_ = tflite::GetModel(g_fatigue_model_data);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema mismatch!");
        return false;
    }

    static tflite::MicroMutableOpResolver<8> resolver;

    if (resolver.AddConv2D() != kTfLiteOk) return false;
    if (resolver.AddDepthwiseConv2D() != kTfLiteOk) return false;
    //if (resolver.AddReshape() != kTfLiteOk) return false;
    if (resolver.AddFullyConnected() != kTfLiteOk) return false; // Dense layer
    if (resolver.AddLogistic() != kTfLiteOk)       return false; // Sigmoid activation
    if (resolver.AddAveragePool2D() != kTfLiteOk)  return false; // Global Average Pooling
    if (resolver.AddAdd() != kTfLiteOk)            return false; // Residual connections
    if (resolver.AddMean() != kTfLiteOk)           return false;
    
    // Build the structural Micro Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model_, resolver, tensor_arena_, kTensorArenaSize);
    interpreter_ = &static_interpreter;

    // Allocate continuous memory across the Arena space
    TfLiteStatus allocate_status = interpreter_->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("Tensor Arena Allocation failed!");
        return false;
    }

    // Capture input/output hooks
    input_tensor_ = interpreter_->input(0);
    output_tensor_ = interpreter_->output(0);

    return true;
}

BBox FatigueClassifier::GetFeatureBBox(const LandmarkPoint* landmarks, const int* indices, 
                                       int index_count, int frame_w, int frame_h, int padding) {
    int x_min = frame_w, y_min = frame_h, x_max = 0, y_max = 0;

    for (int i = 0; i < index_count; ++i) {
        int idx = indices[i];
        int px = static_cast<int>(landmarks[idx].x * frame_w);
        int py = static_cast<int>(landmarks[idx].y * frame_h);

        if (px < x_min) x_min = px;
        if (px > x_max) x_max = px;
        if (py < y_min) y_min = py;
        if (py > y_max) y_max = py;
    }

    BBox box;
    box.x_min = std::max(0, x_min - padding);
    box.y_min = std::max(0, y_min - padding);
    box.x_max = std::min(frame_w, x_max + padding);
    box.y_max = std::min(frame_h, y_max + padding);
    return box;
}

void FatigueClassifier::CropAndResizePatch(const uint8_t* src_frame, int src_w, int src_h, 
                                           BBox box, uint8_t* dest_patch, int dest_w, int dest_h) {
    int box_w = box.x_max - box.x_min;
    int box_h = box.y_max - box.y_min;

    if (box_w <= 0 || box_h <= 0) return;

    // Fixed-point Bilinear Resampling implementation optimized for microcontrollers (No floats)
    for (int y = 0; y < dest_h; ++y) {
        for (int x = 0; x < dest_w; ++x) {
            int src_x = box.x_min + (x * box_w) / dest_w;
            int src_y = box.y_min + (y * box_h) / dest_h;

            // Assuming a standard RGB888 interleaved array from the camera hardware
            int src_idx = (src_y * src_w + src_x) * 3;
            int dest_idx = (y * dest_w + x) * 3;

            dest_patch[dest_idx]     = src_frame[src_idx];     // R
            dest_patch[dest_idx + 1] = src_frame[src_idx + 1]; // G
            dest_patch[dest_idx + 2] = src_frame[src_idx + 2]; // B
        }
    }
}

bool FatigueClassifier::PreprocessAndPack(const uint8_t* frame_buffer, int frame_w, int frame_h, 
                                          const LandmarkPoint* landmarks, int landmark_count) {
    
    // 1. Calculate the local targeted spatial boundaries
    BBox r_eye_box = GetFeatureBBox(landmarks, RIGHT_EYE_INDICES, 6, frame_w, frame_h, 10);
    BBox l_eye_box = GetFeatureBBox(landmarks, LEFT_EYE_INDICES, 6, frame_w, frame_h, 10);
    BBox mouth_box = GetFeatureBBox(landmarks, MOUTH_INDICES, 8, frame_w, frame_h, 10);

    // Temp storage buffers inside stacks instead of dynamic heap pointers to prevent leaks
    uint8_t r_eye_patch[48 * 48 * 3];
    uint8_t l_eye_patch[48 * 48 * 3];
    uint8_t mouth_patch[96 * 48 * 3];

    // 2. Perform the isolated cropping operations
    CropAndResizePatch(frame_buffer, frame_w, frame_h, r_eye_box, r_eye_patch, 48, 48);
    CropAndResizePatch(frame_buffer, frame_w, frame_h, l_eye_box, l_eye_patch, 48, 48);
    CropAndResizePatch(frame_buffer, frame_w, frame_h, mouth_box, mouth_patch, 96, 48);

    // 3. Dense Stitched Vector Compositing directly into the input tensor buffer (96x96x3)
    int8_t* tensor_input_ptr = input_tensor_->data.int8;
    float input_scale = input_tensor_->params.scale;
    int32_t input_zero_point = input_tensor_->params.zero_point;

    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            uint8_t r = 0, g = 0, b = 0;

            if (y < 48) { // Top half assembly: Eyes side by side[cite: 2]
                if (x < 48) {
                    int idx = (y * 48 + x) * 3;
                    r = r_eye_patch[idx]; g = r_eye_patch[idx+1]; b = r_eye_patch[idx+2];
                } else {
                    int idx = (y * 48 + (x - 48)) * 3;
                    r = l_eye_patch[idx]; g = l_eye_patch[idx+1]; b = l_eye_patch[idx+2];
                }
            } else { // Bottom half assembly: Mouth patch[cite: 2]
                int idx = ((y - 48) * 96 + x) * 3;
                r = mouth_patch[idx]; g = mouth_patch[idx+1]; b = mouth_patch[idx+2];
            }

            // Grayscale transformation trick matching Python side configuration[cite: 2]
            uint8_t gray = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);

            // Replicate Gray across 3 channels, scale normalize (0.0 - 1.0f) and convert to INT8[cite: 2]
            float normalized_val = static_cast<float>(gray) / 255.0f;
            int8_t quantized_val = static_cast<int8_t>(normalized_val / input_scale + input_zero_point);

            int tensor_pixel_idx = (y * 96 + x) * 3;
            tensor_input_ptr[tensor_pixel_idx]     = quantized_val;
            tensor_input_ptr[tensor_pixel_idx + 1] = quantized_val;
            tensor_input_ptr[tensor_pixel_idx + 2] = quantized_val;
        }
    }
    return true;
}

float FatigueClassifier::RunInference() {
    // Invoke the Interpreter onto the hardware layers[cite: 2]
    if (interpreter_->Invoke() != kTfLiteOk) {
        MicroPrintf("Inference Engine Invoke Fault!");
        return -1.0f;
    }

    // Unpack INT8 quantized raw layer scalar value into Dequantized Standard Probabilities[cite: 2]
    int8_t quantized_output = output_tensor_->data.int8[0];
    float output_scale = output_tensor_->params.scale;
    int32_t output_zero_point = output_tensor_->params.zero_point;

    float fatigue_probability = (static_cast<float>(quantized_output) - output_zero_point) * output_scale;
    return fatigue_probability;
}