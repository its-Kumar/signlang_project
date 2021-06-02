import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image as image_utils

# * GLOBALS
# Alphabet does not contain j or z because they require movement
ALPHABETS = "abcdefghiklmnopqrstuvwxy"
dictionary = {}
for i in range(24):
    dictionary[i] = ALPHABETS[i]

# Load saved model
model = keras.models.load_model('../models/3_sl_augmented_model')


# Show the image
def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')


# Loading and scaling the image
def load_and_scale_image(image_path):
    image = image_utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(28, 28)
    )
    return image


# Predicting the letter
def predict_letter(file_path):
    # Show image
    # // show_image(file_path)
    # Load and scale image
    image = load_and_scale_image(file_path)
    # Convert to array
    image = image_utils.img_to_array(image)
    # Reshape image
    image = image.reshape(1, 28, 28, 1)
    # Normalize image
    image = image / 255
    # Make prediction
    prediction = model.predict(image)
    # Convert prediction to letter
    predicted_letter = dictionary[np.argmax(prediction)]
    # Return prediction
    return predicted_letter


if __name__ == "__main__":
    image_path = '../data/asl_images/a.png'
    # // show_image(image_path)
    print(predict_letter(image_path))
