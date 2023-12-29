from rembg import remove
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
import os
import time

def remove_background(input_path, output_path):
    # Load the input image
    input_image = Image.open(input_path)

    # Use rembg to remove the background
    output_image = remove(input_image)

    # Create a new image with a black background
    new_image = Image.new("RGBA", output_image.size, (0, 0, 0, 255))

    # Paste the transparent image onto the black background
    new_image.paste(output_image, (0, 0), output_image)

    # Save the final image with a black background
    new_image.save(output_path)

startDetection = time.time()

# Specify the paths
input_path = 'hand_without_landmarks.png'
output_path = 'hand_without_landmarks_clear.png'

# Call the background removal function
remove_background(input_path, output_path)

# Load the trained model
model = load_model('als_model.h5')

test_image_path = output_path  # Use the output of background removal as the test image

img_size = (400, 400)

# Load the test image
test_img = cv2.imread(test_image_path)
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
