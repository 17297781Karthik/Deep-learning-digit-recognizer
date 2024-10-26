import cv2
import numpy as np
from main import model
input_image_path = input('Enter the path of the image to be predicted: ')

# Read the image
input_image = cv2.imread(input_image_path)

# Convert the image to grayscale
grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Resize the image to 28x28
input_image_resize = cv2.resize(grayscale, (28, 28))

# Normalize the image
input_image_resize = input_image_resize / 255.0

# Reshape the image for the model
image_reshaped = np.reshape(input_image_resize, [1, 28, 28])

# Make a prediction
input_prediction = model.predict(image_reshaped)

# Get the predicted label
input_pred_label = np.argmax(input_prediction)

# Print the result
print('The Handwritten Digit is recognised as', input_pred_label)
