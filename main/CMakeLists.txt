idf_component_register(
    SRCS 
        "main.cpp" 
        "heuristic.cpp" 
        "deep_learning.cpp"
        "fatigue_model_data.cc"
    INCLUDE_DIRS 
        "."
    REQUIRES 
        esp-tflite-micro
        espressif__esp32-camera
        freertos
        driver
        esp_timer
        espressif__esp-dl
)
target_compile_options(${COMPONENT_LIB} PRIVATE "-Wno-error=format")