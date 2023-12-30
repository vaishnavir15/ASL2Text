import testDetectionCombine as TD
import time

start_detection = time.time()

TD.capture_hand_image()

end_detection = time.time()
total_detection = end_detection - start_detection
# print(f"Time taken for background removal and prediction: {total_detection} seconds")
print("DONE")