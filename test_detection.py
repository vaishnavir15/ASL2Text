import cv2
import numpy as np
from keras.models import load_model
import os
import time

startDetection = time.time()
# Load the trained model
model = load_model('als_model.h5')

test_image_path = 'images/hand_without_landmarks-PhotoRoom.png-PhotoRoom.png'  # Replace with the path to your test image 
img_size = (400, 400)

test_img = cv2.imread(test_image_path)
print(test_img)
if test_img is None:
    print(f"Error loading image from path: {test_image_path}")
else:
    print("Image loaded successfully.")
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, img_size)
    test_img = test_img / 255.0  # Normalize pixel values to be between 0 and 1
    test_img = np.expand_dims(test_img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(test_img)

    # Assuming it's a classification task, get the predicted class
    predicted_class = np.argmax(predictions)

    # Print or use the predicted class as needed
    print(f"Predicted Class: {predicted_class}")

endDetection = time.time()
totalDetection = endDetection - startDetection
print(f"Time taken to train model: {totalDetection} seconds")
print("DONE")

