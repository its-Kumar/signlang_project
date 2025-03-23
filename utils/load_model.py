"""Load model from the disk."""

import logging

import tensorflow as tf
from tensorflow import keras

from .conf import BASE_DIR

logger = logging.getLogger(__name__)
tf.config.set_visible_devices([], "GPU")


def load_saved_model(model_name: str):
    """Load saved keras model from the disk.

    Args:
        model_path (str): name of the model

    Returns:
        keras.Model | None: Loaded model or None if an error occurs

    """
    try:
        path = BASE_DIR / "models" / model_name
        logger.info("Loading model: %s .....", path)
        return keras.models.load_model(path)
    except ImportError:
        logger.exception(
            "An error occurred while loading the model, make sure the model has valid format")
        return None
    except OSError:
        logger.exception(
            "An error occurred while loading the model, make sure to provide the correct path")
        return None


if __name__ == "__main__":
    model = load_saved_model("2_sl_cnn_model")
    if model:
        print("Model loaded successfully")
