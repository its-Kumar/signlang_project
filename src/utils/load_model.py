"""Load model from the disk"""


from tensorflow import keras

from .conf import BASE_DIR


def load_saved_model(model_path: str):
    """
    Load saved keras model from the disk

    Args:
        model_path (str): path of the model
    """
    path = BASE_DIR / 'models' / model_path
    print(f"Loading model: {path} .....")
    model = keras.models.load_model(path)
    return model


if __name__ == "__main__":
    model = load_saved_model('2_sl_cnn_model')
