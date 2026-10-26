#ifndef FATIGUE_MODEL_DATA_H
#define FATIGUE_MODEL_DATA_H

// Alignment attribute required for the ESP32-S3 to read arrays directly from Flash
#if defined(__GNUC__)
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(16)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

// Expose the global raw model byte-array to other compilation units
extern const unsigned char g_fatigue_model_data[] DATA_ALIGN_ATTRIBUTE;
extern const int g_fatigue_model_data_len;

#endif // FATIGUE_MODEL_DATA_H