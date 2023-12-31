import cv2
import numpy as np
from keras.models import load_model

def predict_result( output_path, model_path):
    # Load the trained model, resize and normalize the image, then predict it using the preloaded model
    model = load_model(model_path)

    test_img = cv2.imread(output_path)
    if test_img is None:
        print(f"Error loading image from path: {output_path}")
        return None

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (400, 400))
    test_img = test_img / 255.0
    test_img = np.expand_dims(test_img, axis=0)

    predictions = model.predict(test_img)

    predicted_class = np.argmax(predictions)

    return predicted_class
