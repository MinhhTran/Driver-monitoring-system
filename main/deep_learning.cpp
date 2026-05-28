#include "deep_learning.h"
#include "fatigue_model_data.h"
#include "esp_heap_caps.h"
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
    tensor_arena_ = (uint8_t*)heap_caps_aligned_alloc(
        16, // Strict 16-Byte alignment for TFLite Micro SIMD operations
        kTensorArenaSize, 
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    if (tensor_arena_ == nullptr) {
        MicroPrintf("FATAL: Failed to allocate Tensor Arena in PSRAM!");
        return false;
    }
    
    interpreter_ = new tflite::MicroInterpreter(
        model_, resolver, tensor_arena_, kTensorArenaSize
    );

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

BBox FatigueClassifier::GetBoxAroundCenter(LandmarkPoint center, int width, int height, 
                                           int frame_w, int frame_h) {
    BBox b;
    // Convert normalized coordinates back to absolute pixel values
    int cx = static_cast<int>(center.x * frame_w);
    int cy = static_cast<int>(center.y * frame_h);

    // Create the box and clamp to image boundaries to prevent SegFaults
    b.x_min = std::max(0, cx - (width / 2));
    b.y_min = std::max(0, cy - (height / 2));
    b.x_max = std::min(frame_w, cx + (width / 2));
    b.y_max = std::min(frame_h, cy + (height / 2));
    return b;
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
                                          const MTMNFace& face) {
    // 1. Calculate dynamic feature sizes based on the detected face size
    int face_width = face.box.x_max - face.box.x_min;
    int face_height = face.box.y_max - face.box.y_min;

    // Estimate: Eye box is ~35% of face width, Mouth is ~50% width and 30% height
    int eye_size = static_cast<int>(face_width * 0.35f);
    int mouth_w = static_cast<int>(face_width * 0.50f);
    int mouth_h = static_cast<int>(face_height * 0.30f);

    // 2. Generate the bounding boxes around the MTMN keypoints
    // keypoint[0] for Left Eye and keypoint[1] for Right Eye
    BBox l_eye_box = GetBoxAroundCenter(face.left_eye, eye_size, eye_size, frame_w, frame_h);
    BBox r_eye_box = GetBoxAroundCenter(face.right_eye, eye_size, eye_size, frame_w, frame_h);

    // For the mouth, the center is the midpoint between the left and right mouth corners
    LandmarkPoint mouth_center;
    mouth_center.x = (face.left_mouth.x + face.right_mouth.x) / 2.0f;
    mouth_center.y = (face.left_mouth.y + face.right_mouth.y) / 2.0f;
    BBox mouth_box = GetBoxAroundCenter(mouth_center, mouth_w, mouth_h, frame_w, frame_h);

    // Temp storage buffers (stacks allocated)
    uint8_t* r_eye_patch = (uint8_t*)malloc(48 * 48 * 3);
    uint8_t* l_eye_patch = (uint8_t*)malloc(48 * 48 * 3);
    uint8_t* mouth_patch = (uint8_t*)malloc(96 * 48 * 3);

    if (!r_eye_patch || !l_eye_patch || !mouth_patch) {
        if (r_eye_patch) free(r_eye_patch);
        if (l_eye_patch) free(l_eye_patch);
        if (mouth_patch) free(mouth_patch);
        return false;
    }

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
    free(r_eye_patch);
    free(l_eye_patch);
    free(mouth_patch);
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