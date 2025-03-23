import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as image_utils

from .conf import BASE_DIR, dictionary
from .load_model import load_saved_model


# Show the image
def show_image(image_path: str) -> None:
    """Show the image using the image path.

    Args:
        image_path (str): image path

    """
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap="gray")
    plt.show()


# Loading and scaling the image
def load_and_scale_image(image_path: str) -> Image.Image:
    """Load and scale the image for prediction.

    Args:
        image_path (str): image path

    Returns:
        Image.Image: image object

    """
    image = image_utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(28, 28)
    )
    return image


def predict_letter(model: object, image: Image.Image) -> str:
    """Predict the letter from the image.

    Args:
        model (object): TensorFlow model
        image (Image.Image): Image for prediction

    Returns:
        str: Predicted letter

    """
    # Reshape image
    image = image.reshape(1, 28, 28, 1)
    # Normalize image
    image = image / 255
    # Make prediction
    prediction = model.predict(image)
    # Convert prediction to letter
    return dictionary[np.argmax(prediction)]


if __name__ == "__main__":
    image_path = BASE_DIR / "data/asl_images/a.png"

    image = load_and_scale_image(image_path)
    image = np.array(image, dtype=np.uint8)
    model = load_saved_model("3_sl_aug_model")
    print("Predicted Letter: ", predict_letter(model, image))
