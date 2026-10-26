import numpy as np
import cv2

# --- CONFIGURATION ---
# Set to False if viewing RIGHT_EYE_PATCH. Set to True if viewing FINAL_TENSOR.
IS_FINAL_TENSOR = True
with open("hex_dump_1.txt", "r") as f:
    hex_str = f.read().strip()

byte_array = bytes.fromhex(hex_str)

if IS_FINAL_TENSOR:
    # Final Tensor is 96x96x3, Quantized INT8
    image_array = np.frombuffer(byte_array, dtype=np.int8)
    image_array = ((image_array.astype(np.float32) + 128)).astype(np.uint8)
    image = image_array.reshape((96, 96, 3)) # Reshape to 96x96 RGB
else:
    # Right Eye Patch is 48x48x3, standard UINT8
    image_array = np.frombuffer(byte_array, dtype=np.uint8)
    image = image_array.reshape((48, 48, 3)) # Reshape to 48x48 RGB

# Display the image
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
image_large = cv2.resize(image, (400, 400), interpolation=cv2.INTER_NEAREST)
cv2.imshow("ESP32 Buffer", image_large)
cv2.waitKey(0)