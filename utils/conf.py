"""Common configuraions for model."""

import pathlib

# * GLOBALS
BASE_DIR = pathlib.Path(__file__).parent.parent


# Alphabet does not contain j or z because they require movement
ALPHABETS = "abcdefghiklmnopqrstuvwxy"
dictionary = {}
for i in range(24):
    dictionary[i] = ALPHABETS[i]
