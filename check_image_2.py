import numpy as np
import cv2

# The ESP32 camera usually defaults to QVGA (320x240)
WIDTH = 320
HEIGHT = 240

with open("hex_dump_2.txt", "r") as f:
    hex_str = f.read().strip()

# Convert hex to bytes
byte_array = bytes.fromhex(hex_str)

# Convert to Numpy Array
image_array = np.frombuffer(byte_array, dtype=np.uint8)

try:
    # Reshape to 320x240 RGB
    image = image_array.reshape((HEIGHT, WIDTH, 3))
    
    # ESP32 usually puts out RGB, OpenCV needs BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Make it bigger on your laptop screen
    image_large = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("Raw ESP32 Camera Output", image_large)
    cv2.waitKey(0)
except Exception as e:
    print(f"Reshape failed: {e}. Expected {WIDTH*HEIGHT*3} bytes, but got {len(byte_array)} bytes.")
    print("Check if your camera resolution is actually 320x240!")