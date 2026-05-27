#include <stdio.h>
#include <vector>
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"

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
    config.frame_size = FRAMESIZE_QVGA; // 320x240
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
}

// ==========================================
// Main Execution Loop
// ==========================================
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

        // 2. FACE DETECTION & LANDMARK EXTRACTION
        // ---------------------------------------------------------
        // NOTE: You must insert your ESP-WHO / MTMN face detection 
        // inference here. It should parse the `fb->buf` and return 
        // the 468 (or subset) facial landmarks.
        // ---------------------------------------------------------
        std::vector<Point> heuristic_landmarks; // Populate via MTMN
        std::vector<LandmarkPoint> dl_landmarks; // Populate via MTMN
        bool face_detected = false; // Set to true if MTMN finds a face

        if (face_detected && heuristic_landmarks.size() > 0) {
            
            // 3. Run Pipeline A: Heuristic (Geometric)
            heuristic_engine.UpdateMetrics(heuristic_landmarks);
            bool is_drowsy_heuristic = heuristic_engine.IsDrowsy();
            bool is_yawning = heuristic_engine.IsYawning();

            // 4. Run Pipeline B: Deep Learning (TFLite Micro)
            bool pack_success = dl_classifier.PreprocessAndPack(
                fb->buf, fb->width, fb->height, 
                dl_landmarks.data(), dl_landmarks.size()
            );

            float fatigue_prob = 0.0f;
            if (pack_success) {
                fatigue_prob = dl_classifier.RunInference();
            }

            // 5. Sensor Fusion & Alert Logic
            // Example: Trigger alert if either pipeline detects high fatigue
            if (is_drowsy_heuristic || is_yawning || fatigue_prob > 0.75f) {
                ESP_LOGW(TAG, "FATIGUE DETECTED! EAR/MAR threshold or ML Prob: %.2f", fatigue_prob);
                gpio_set_level(ALERT_PIN, 1); // Trigger Buzzer
            } else {
                gpio_set_level(ALERT_PIN, 0); // Turn off Buzzer
            }
        } else {
            // Turn off alerts if no face is detected to prevent false positives
            gpio_set_level(ALERT_PIN, 0);
        }

        // Return the frame buffer back to the camera driver
        esp_camera_fb_return(fb);

        // Yield to allow other FreeRTOS tasks to run
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }
}

// ==========================================
// Application Entry Point
// ==========================================
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

    // Pin the DMS task to Core 1 (Core 0 handles Wi-Fi/System by default)
    BaseType_t task_status = xTaskCreatePinnedToCore(
        dms_task, 
        "DMS_Task", 
        4096 * 4, // 16KB Stack size, adjust if MTMN needs more
        NULL, 
        5, 
        NULL, 
        1 // Core 1
    );

    if (task_status != pdPASS) {
        ESP_LOGE(TAG, "DMS Task Creation Failed! Error Code: %d (Out of SRAM Heap)", task_status);
    } else {
        ESP_LOGI(TAG, "DMS Task Successfully Spawned.");
    }
}