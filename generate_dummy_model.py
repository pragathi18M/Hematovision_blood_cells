import os, pathlib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 1️⃣  Build a tiny network (224×224 RGB → 4-class softmax)
def build_dummy(img_h=224, img_w=224, n_classes=4):
    return Sequential([
        Flatten(input_shape=(img_h, img_w, 3)),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])

# 2️⃣  Make sure model/ exists and save the .h5 file there
path = pathlib.Path("model")
path.mkdir(exist_ok=True)
build_dummy().save(path / "model.h5")
print("✅ Dummy model saved to", path / "model.h5")
