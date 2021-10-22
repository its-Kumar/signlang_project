
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image as image_utils

from .conf import BASE_DIR, dictionary
from .load_model import load_saved_model


# Show the image
def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')
    plt.show()


# Loading and scaling the image
def load_and_scale_image(image_path):
    image = image_utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(28, 28)
    )
    return image


# Predicting the letter
def predict_letter(model, image):
    # Show image
    # // show_image(file_path)
    # Load and scale image
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
    image_path = BASE_DIR / 'data/asl_images/a.png'
    # show_image(image_path)
    image = load_and_scale_image(image_path)
    image = np.array(image, dtype=np.uint8)
    model = load_saved_model('3_sl_aug_model')
    print(predict_letter(model, image))
