#include <stdio.h>
#include <vector>
#include <list>
#include <inttypes.h>
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "img_converters.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "dl_image.hpp"
#include "human_face_detect_msr01.hpp"
#include "human_face_detect_mnp01.hpp"
#include "heuristic.h"
#include "deep_learning.h"

static const char *TAG = "DMS_MAIN";

// ==========================================
// Hardware Configuration (XIAO ESP32-S3 Sense)
// ==========================================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// Alert Pin (e.g., Active Buzzer or LED)
#define ALERT_PIN GPIO_NUM_21

// ==========================================
// Global Pipeline Objects
// ==========================================
HeuristicPipeline heuristic_engine;
FatigueClassifier dl_classifier;

// ==========================================
// Camera Initialization
// ==========================================
static esp_err_t init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565; 
    config.frame_size = FRAMESIZE_SVGA; // 320x240
    config.jpeg_quality = 12;
    config.fb_count = 2; // Requires PSRAM
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        return err;
    }
    return ESP_OK;

    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        ESP_LOGI(TAG, "Applying OV3660 tuning for AI detection");
        
        // 1. Lighting and Glare Control
        s->set_brightness(s, 2);     // +2: Brighten to reveal eyes behind glasses frames
        s->set_contrast(s, -1);      // -1: Soften harsh shadows that trick the model
        s->set_saturation(s, -1);    // -1: Flatten colors (AI models prefer this)
        
        // 2. Ensure Auto-Exposure is handling cabin light changes
        s->set_exposure_ctrl(s, 1);  // 1 = Enable auto exposure
        s->set_aec2(s, 1);           // 1 = Enable advanced DSP auto exposure
        s->set_gain_ctrl(s, 1);      // 1 = Enable auto gain
        // s->set_special_effect(s, 2); // 2 = Grayscale
    } else {
        ESP_LOGW(TAG, "Failed to get camera sensor object for tuning.");
    }

    return ESP_OK;
}

// ==========================================
// Main Execution Loop
// ==========================================
//#define RUN_HEURISTIC_PIPELINE
#define RUN_DEEP_LEARNING_PIPELINE

HumanFaceDetectMSR01 face_detector_stage1(0.2F, 0.3F, 1, 0.3F);
HumanFaceDetectMNP01 face_detector_stage2(0.3F, 0.3F, 1);

void dms_task(void *pvParameters) {
    ESP_LOGI(TAG, "DMS Task Started");
    while (true) {
        // 1. Capture Camera Frame
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        //ESP_LOGI(TAG, "Frame captured! Resolution: %dx%d, Size: %zu bytes", fb->width, fb->height, fb->len);

        uint8_t *rgb888_buf = (uint8_t *)heap_caps_malloc(fb->width * fb->height * 3, MALLOC_CAP_SPIRAM);
        bool face_detected = false;
        MTMNFace detected_face;

        if (rgb888_buf) {
            // Convert raw camera format RGB565 to RGB888
            fmt2rgb888(fb->buf, fb->len, fb->format, rgb888_buf);

            // 2. FACE DETECTION & LANDMARK EXTRACTION
            // Run ESP-DL inference on the converted image
            //std::list<dl::detect::result_t> &results = face_detector.infer(
            //    rgb888_buf, {fb->height, fb->width, 3}
            //);
            static int frames_since_last_detection = 0;
            const int MAX_TRACKING_FRAMES = 10; // At ~10 FPS, this is 1 second of yawning/glare forgiveness
            static dl::detect::result_t last_known_face; 
            static bool has_last_known_face = false;

            auto &candidates = face_detector_stage1.infer(rgb888_buf, {(int)fb->height, (int)fb->width, 3});
            auto &results = face_detector_stage2.infer(rgb888_buf, {(int)fb->height, (int)fb->width, 3}, candidates);
            ESP_LOGI(TAG, "Inference complete. Faces found: %d", results.size());
            dl::detect::result_t best_face;
            if (!results.empty()) {
                face_detected = true;
                best_face = results.front();
                last_known_face = best_face;
                has_last_known_face = true;
                frames_since_last_detection = 0;
                //ESP_LOGI(TAG, "MTMN Box: [%f, %f, %f, %f]", best_face.box[0], best_face.box[1], best_face.box[2], best_face.box[3]);
                //ESP_LOGI(TAG, "MTMN Keypoint Array Size: %d", best_face.keypoint.size());

                //if (best_face.keypoint.size() >= 10) {
                //    ESP_LOGI(TAG, "Left Eye X: %f", best_face.keypoint[0]);
                //} else {
                //    ESP_LOGE(TAG, "CRITICAL: MTMN did not output keypoints!");
                //}
            } else {
                if (has_last_known_face && frames_since_last_detection < MAX_TRACKING_FRAMES) {
                    ESP_LOGW(TAG, "Face lost (glare/yawn). Using last known pos (Frame %d/%d)", frames_since_last_detection + 1, MAX_TRACKING_FRAMES);
                    face_detected = true;
                    best_face = last_known_face; // Restore the cached bounding box & keypoints
                    frames_since_last_detection++;
                } else {
                    has_last_known_face = false; // Reset the cache just to be safe
                    face_detected = false;
                }
            }

            if (face_detected) {
                if (best_face.keypoint.size() < 10) {
                    ESP_LOGE(TAG, "CRITICAL: MTMN returned invalid keypoint size!");
                    face_detected = false; // Abort this frame
                } else {
                    // Populate our unified MTMNFace struct
                    // A. Absolute Bounding Box
                    detected_face.box.x_min = std::max(0, best_face.box[0]);
                    detected_face.box.y_min = std::max(0, best_face.box[1]);
                    detected_face.box.x_max = std::min((int)fb->width, best_face.box[2]);
                    detected_face.box.y_max = std::min((int)fb->height, best_face.box[3]);

                    // B. Normalized Keypoints (ESP-DL gives absolute, so we convert them)
                    // [0-1] Left Eye, [2-3] Right Eye, [4-5] Nose, [6-7] Left Mouth, [8-9] Right Mouth
                    detected_face.left_eye.x    = best_face.keypoint[6] / (float)fb->width;
                    detected_face.left_eye.y    = best_face.keypoint[7] / (float)fb->height;
                    detected_face.right_eye.x   = best_face.keypoint[0] / (float)fb->width;
                    detected_face.right_eye.y   = best_face.keypoint[1] / (float)fb->height;
                    detected_face.nose.x        = best_face.keypoint[4] / (float)fb->width;
                    detected_face.nose.y        = best_face.keypoint[5] / (float)fb->height;
                    detected_face.left_mouth.x  = best_face.keypoint[8] / (float)fb->width;
                    detected_face.left_mouth.y  = best_face.keypoint[9] / (float)fb->height;
                    detected_face.right_mouth.x = best_face.keypoint[2] / (float)fb->width;
                    detected_face.right_mouth.y = best_face.keypoint[3] / (float)fb->height;

                    //ESP_LOGI(TAG, "Running inference pipeline components...");
                    int64_t start_time = esp_timer_get_time();
                    // 3. Run Pipeline A: Heuristic (Geometric)
                    #ifdef RUN_HEURISTIC_PIPELINE
                        heuristic_engine.UpdateMetrics(rgb888_buf, fb->width, fb->height, detected_face);
                        bool is_drowsy_heuristic = heuristic_engine.IsDrowsy();
                        bool is_yawning = heuristic_engine.IsYawning();

                        if (is_drowsy_heuristic || is_yawning > 0.75f) {
                            ESP_LOGW(TAG, "FATIGUE DETECTED!");
                            gpio_set_level(ALERT_PIN, 1); // Trigger Buzzer
                        } else {
                            gpio_set_level(ALERT_PIN, 0); // Turn off Buzzer
                        }

                        int64_t end_time = esp_timer_get_time();
                        float latency_ms = (end_time - start_time) / 1000.0f;
                        size_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
                        size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

                        ESP_LOGI(TAG, "METRICS | EAR: %.3f | MAR: %.3f | Latency: %.2f ms | Free SRAM: %zu B | Free PSRAM: %zu B",
                            heuristic_engine.GetEAR(),
                            heuristic_engine.GetMAR(),
                            latency_ms,
                            free_sram,
                            free_psram);
                    #endif

                    // 4. Run Pipeline B: Deep Learning (TFLite Micro)
                    #ifdef RUN_DEEP_LEARNING_PIPELINE
                        bool pack_success = dl_classifier.PreprocessAndPack(
                            rgb888_buf, fb->width, fb->height, detected_face
                        );

                        float fatigue_prob = 0.0f;
                        if (pack_success) {
                            fatigue_prob = dl_classifier.RunInference();
                        }

                        if (fatigue_prob > 0.7f) {
                            ESP_LOGW(TAG, "FATIGUE DETECTED! ML Prob: %.2f", fatigue_prob);
                            gpio_set_level(ALERT_PIN, 1); // Trigger Buzzer
                        } else {
                            gpio_set_level(ALERT_PIN, 0); // Turn off Buzzer
                        }

                        int64_t end_time = esp_timer_get_time();
                        float latency_ms = (end_time - start_time) / 1000.0f;
                        size_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
                        size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

                        ESP_LOGI(TAG, "METRICS | Fatigue: %.1f%% | Latency: %.2f ms | Free SRAM: %zu B | Free PSRAM: %zu B",
                                fatigue_prob * 100.0f,
                                latency_ms,
                                free_sram,
                                free_psram);
                    #endif
                }
            } else {
                gpio_set_level(ALERT_PIN, 0);
            }
        } else {
            ESP_LOGE(TAG, "Failed to allocate RGB888 buffer! Check SRAM availability.");
        }
        
        // Free image buffer
        if (rgb888_buf) {
            free(rgb888_buf);
        }
        // Return the frame buffer back to the camera driver
        esp_camera_fb_return(fb);

        // Yield to allow other FreeRTOS tasks to run
        vTaskDelay(pdMS_TO_TICKS(30)); 
    }
}

// ==========================================
// Application Entry Point
// ==========================================
static StaticTask_t dms_task_buffer;
static StackType_t* dms_task_stack = nullptr;
const uint32_t dms_stack_size = 8192 * 4;

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Starting ESP32-S3 Driver Monitoring System");

    // Initialize Alert GPIO
    gpio_reset_pin(ALERT_PIN);
    gpio_set_direction(ALERT_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(ALERT_PIN, 0);

    // Initialize Camera
    if (init_camera() != ESP_OK) {
        ESP_LOGE(TAG, "Halting due to camera failure.");
        return;
    }

    // Initialize Deep Learning Classifier
    if (!dl_classifier.Init()) {
        ESP_LOGE(TAG, "TFLite Micro Classifier Initialization Failed!");
        return;
    }
    ESP_LOGI(TAG, "TFLite Micro loaded successfully.");

    dms_task_stack = (StackType_t*)heap_caps_malloc(dms_stack_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    if (dms_task_stack == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate 32KB DMS task stack in PSRAM!");
        return;
    }

    // Pin the DMS task to Core 1 (Core 0 handles Wi-Fi/System by default)
    TaskHandle_t dms_task_handle = xTaskCreateStaticPinnedToCore(
        dms_task,
        "DMS_Task",
        dms_stack_size,
        NULL,
        5,               // Priority level
        dms_task_stack,  // Pointer to the memory buffer in PSRAM
        &dms_task_buffer, // Pointer to the task control architecture block
        1                // Assigned to Core 1
    );

    if (dms_task_handle == nullptr) {
        ESP_LOGE(TAG, "Static DMS Task Creation Failed!");
    } else {
        ESP_LOGI(TAG, "DMS Task Successfully Allocated and Spawned in PSRAM.");
    }
}