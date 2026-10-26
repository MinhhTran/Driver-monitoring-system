# Embedded Real-Time Driver Monitoring System (DMS)
### Comparative Analysis of Geometric Heuristics vs. Quantized TinyML on the ESP32-S3

[![Platform: ESP32-S3](https://img.shields.io/badge/Platform-ESP32--S3-red.svg)](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html)
[![Framework: ESP-IDF / FreeRTOS](https://img.shields.io/badge/Framework-ESP--IDF%20%2F%20FreeRTOS-blue.svg)](https://github.com/espressif/esp-idf)
[![ML: TensorFlow Lite Micro](https://img.shields.io/badge/TinyML-TFLite%20Micro%20(INT8)-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)

An edge-deployed driver drowsiness and micro-sleep monitoring system running entirely on-device without cloud connectivity. Developed as part of a Bachelor’s Thesis at Aalto University, this project evaluates the performance, latency, and noise resilience trade-offs between a classical geometric signal processing pipeline (EAR/MAR + 1D Kalman Filter) and an optimized deep learning pipeline (Post-Training Quantized MobileNetV2 on TFLite Micro).

---

## 🚀 Key Highlights & Architectural Insights

* **Strict Edge Privacy & Zero Cloud Dependency:** All video acquisition, facial landmarking, inference, and alert actuation execute on-chip.
* **TinyML Quantization:** Compressed a 32-bit floating-point MobileNet model (5,200 KB) down to 600 KB INT8 weights via Post-Training Quantization (PTQ) to fit embedded constraints.
* **Dual-Memory Layout Optimization:** Mitigated ESP32-S3 internal SRAM heap fragmentation by dynamically routing camera RGB565/RGB888 frame buffers (~230 KB) and the TFLite Micro Tensor Arena (~224 KB) to external 8 MB Octal PSRAM, preserving internal 512 KB SRAM for the FreeRTOS scheduler and core math.
* **Key Empirical Finding:** While geometric heuristics achieve near-instantaneous execution on PC prototypes (using dense 468-point MediaPipe meshes), they break down when paired with lightweight edge detectors (5-point MTMN) due to keypoint jitter. The quantized CNN proved vastly superior in adverse conditions by evaluating holistic spatial pixel features rather than brittle Euclidean distances.

---

## 🛠️ System Architecture
```text
                 +-------------------------------+
                 |  OV3660 Camera (RGB565 frame) |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 |   MTMN Face & Landmark Detect  |
                 |      (MSR01 / MNP01 Models)   |
                 +---------------+---------------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
+-------------------------------+       +-------------------------------+
| Pipeline A: Heuristic + Kalman|       | Pipeline B: Quantized TinyML  |
| • 1D Kalman (Keypoints)       |       | • Crop & Stitch 96x96 Input   |
| • Euclidean Geometry (EAR/MAR)|       | • MobileNetV2 INT8 Model      |
| • 1D Kalman (Signal Smooth)   |       | • TFLite Micro in PSRAM       |
| • PERCLOS Sliding Window (60s)|       | • State Confidence > 70%      |
+---------------+---------------+       +---------------+---------------+
        |                                                 |
        +------------------------+------------------------+
                                 |
                                 v
                 +-------------------------------+
                 |  State Machine / Alert Logic  |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 | Actuation: GPIO 21 LED/Buzzer |
                 +-------------------------------+
```

## 📊 Experimental Benchmarks & Evaluation

Both pipelines were tested head-to-head on the physical Seeed Studio XIAO ESP32-S3 Sense under controlled adverse conditions (60-second observation window / 1,800 frames per trial).

### 1. Baseline Condition (Standard Daytime, 300–500 Lux, Direct Angle)
| Metric | Pipeline A (Heuristic + Kalman) | Pipeline B (Quantized MobileNetV2) |
| :--- | :---: | :---: |
| **Accuracy** | 66.01% | **95.12%** |
| **True Positive Rate (TPR)** | 0.00% | **91.23%** |
| **False Positive Rate (FPR)** | **0.00%** | 1.52% |
| **Inference / Algorithmic Latency** | **0.32 ms** | 754.97 ms |
| **Total Memory Consumption** | 259.91 KB | 259.84 KB |

### 2. Occlusion Condition (Glasses, Face Mask / Hand Occlusion)
| Metric | Pipeline A (Heuristic + Kalman) | Pipeline B (Quantized MobileNetV2) |
| :--- | :---: | :---: |
| **Accuracy** | 62.88% | **92.74%** |
| **True Positive Rate (TPR)** | 0.00% | **86.05%** |
| **False Positive Rate (FPR)** | **0.00%** | 3.70% |
| **Inference / Algorithmic Latency** | **0.32 ms** | 755.42 ms |
| **Total Memory Consumption** | 259.91 KB | 259.84 KB |

> **Takeaway:** The heuristic pipeline fails to register true positives on the microcontroller because low-cost sensor jitter corrupts the 5-point pupil and lip coordinates. In contrast, the quantized deep learning model satisfies automotive safety baselines (TPR > 90%, FPR < 5%) across both baseline and occluded test scenarios.

---

## 💻 Hardware Bill of Materials

* **MCU Board:** Seeed Studio XIAO ESP32-S3 Sense (Dual-core Xtensa LX7 @ 240MHz, 512KB SRAM, 8MB PSRAM, 8MB Flash)
* **Camera Module:** Omnivision OV3660 (tuned via C++ DSP register settings for brightness and contrast compensation)
* **Actuation:** Active buzzer and visual indicator LED connected to GPIO 21
* **Ground Truth Validation:** External tactile pushbutton interrupt for logging human baseline events
* **Power Supply:** 3.7V 500mAh LiPo battery

---

## 📂 Repository Structure

```text
Driver-monitoring-system/
├── data/                             # Experimental logs and benchmark records
│   ├── dms_final_analysis.csv        # Compiled statistical metrics across test cases
│   ├── dms_performance_log.csv       # Raw runtime logs (latency, accuracy, memory)
│   ├── metrics                       # Consolidated performance metrics summary
│   └── Result                        # Raw trial execution output
│
├── docs/                             # Hardware schematics and physical setup photos
│   └── breadboard_setup.jpg          # XIAO ESP32-S3 + OV3660 test rig photograph
│
├── firmware/                         # Embedded C/C++ project (ESP-IDF & FreeRTOS)
│   ├── main/
│   │   ├── analyze.c                 # Frame evaluation and profiling routines
│   │   ├── CMakeLists.txt            # Main component build script
│   │   ├── deep_learning.cpp         # 96x96 composite matrix stitching & TFLM inference
│   │   ├── deep_learning.h
│   │   ├── fatigue_model_data.cc     # INT8 quantized MobileNetV2 byte array
│   │   ├── fatigue_model_data.h
│   │   ├── heuristic.cpp             # Geometric EAR/MAR calculations & 1D Kalman filter
│   │   ├── heuristic.h
│   │   ├── idf_component.yml         # ESP-IDF component manager (ESP-WHO / MTMN dependencies)
│   │   └── main.cpp                  # FreeRTOS task orchestration, camera init & GPIO actuation
│   ├── CMakeLists.txt                # Project-level CMake definition
│   ├── partitions.csv                # Custom partition table (Flash allocation)
│   └── sdkconfig                     # ESP-IDF hardware config (Octal PSRAM, clock frequency)
│
├── models/                           # Trained neural network weights and landmark models
│   ├── face_landmarker.task          # MediaPipe facial mesh bundle (Python prototype)
│   ├── fatigue_model_base.h5         # Unquantized floating-point MobileNetV2 model
│   └── fatigue_model_quantized.tflite# INT8 post-training quantized model
│
├── training_and_quantization/        # Python pipeline for training, evaluation, and TinyML conversion
│   ├── analyze_metrics.py            # Benchmark visualization and performance plotting
│   ├── camera.py                     # Video stream acquisition and landmark extraction
│   ├── check_image_1.py              # Check camera quality
│   ├── check_image_2.py              # Check camera quality
│   ├── deep_learning.py              # PC-based deep learning inference pipeline
│   ├── filter_data.py                # Dataset filtering and outlier removal
│   ├── heuristic.py                  # PC-based EAR/MAR heuristic and Kalman filter prototype
│   ├── preprocess_data.py            # Facial ROI cropping and input matrix normalization
│   ├── quantize_model.py             # Post-Training Quantization (FP32 -> INT8 TFLite)
│   ├── test_model.py                 # Offline model validation on test datasets
│   └── train_model.py                # MobileNetV2 transfer learning on NTHU-DDD dataset
│
├── .gitignore                        # Git exclusion rules (build artifacts, caches, dumps)
└── dependencies.lock                 # Pinned Python dependencies

```

## 🔧 Getting Started

### Prerequisites
* ESP-IDF v5.1+
* Python 3.9+ with TensorFlow 2.x and OpenCV

### 1. Flash the ESP32-S3 Firmware
```bash
cd firmware
idf.py set-target esp32s3
idf.py menuconfig   # Ensure PSRAM is enabled (Octal mode) and Flash size is set to 8MB
idf.py build
idf.py -p <PORT> flash monitor
```

### 2. Run the Python Prototyping Pipeline
```bash
cd training_and_quantization
pip install -r ../requirements.txt

# Train or evaluate the base model
python train_model.py

# Quantize model to int8 tflite format
python quantize_model.py

```

## 📜 Thesis Reference
This project forms the experimental implementation of the Bachelor of Science Thesis:

Minh Tran, "Facial Landmark Tracking in Challenging Lighting: A Comparative Analysis of Heuristic Versus Deep Learning for Facial Landmark Tracking in Adverse Conditions", Aalto University School of Science, 2026.


***

### Assets to Add

Place the high-resolution breadboard photograph from your project report
