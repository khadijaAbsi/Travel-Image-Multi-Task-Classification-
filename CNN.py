# ------------------
# IMPORT LIBRARIES
# ------------------
import os
import cv2
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")  # stop warning messages

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


# CONFIG

IMAGE_FOLDER = "travel_images"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10


# LOAD & PREPROCESS DATA

images = []
weather_labels = []
time_labels = []

for file in os.listdir(IMAGE_FOLDER):
    if file.endswith((".jpg", ".png", ".jpeg", ".jfif", ".avif", ".webp")):
        try:
            # مثال: Sunny_Morning_001.jpg
            weather, time, _ = file.split("_", 2)

            img = cv2.imread(os.path.join(IMAGE_FOLDER, file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            weather_labels.append(weather)
            time_labels.append(time)
        except:
            pass

X = np.array(images)


# ENCODE LABELS

weather_enc = LabelEncoder()
time_enc = LabelEncoder()

y_weather = to_categorical(weather_enc.fit_transform(weather_labels))
y_time = to_categorical(time_enc.fit_transform(time_labels))


# SPLIT DATA

X_train, X_test, yw_train, yw_test, yt_train, yt_test = train_test_split(
    X, y_weather, y_time, test_size=0.2, random_state=42
)


# BUILD MULTI-TASK CNN

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # freeze pretrained layers

# Shared layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)

# Outputs
weather_output = Dense(y_weather.shape[1], activation="softmax", name="weather")(x)
time_output = Dense(y_time.shape[1], activation="softmax", name="time")(x)

multi_model = Model(
    inputs=base_model.input,
    outputs=[weather_output, time_output]
)


# COMPILE MODEL

multi_model.compile(
    optimizer="adam",
    loss={
        "weather": "categorical_crossentropy",
        "time": "categorical_crossentropy"
    },
    metrics={
        "weather": "accuracy",
        "time": "accuracy"
    }
)


# TRAIN MODEL

multi_model.fit(
    X_train,
    {"weather": yw_train, "time": yt_train},
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# PREDICTION

yw_pred, yt_pred = multi_model.predict(X_test)

yw_pred = np.argmax(yw_pred, axis=1)
yt_pred = np.argmax(yt_pred, axis=1)

yw_true = np.argmax(yw_test, axis=1)
yt_true = np.argmax(yt_test, axis=1)

# EVALUATE 

print("\n=== WEATHER RESULTS ===")
print("Accuracy:", accuracy_score(yw_true, yw_pred))
print(classification_report(
    yw_true,
    yw_pred,
    target_names=weather_enc.classes_,
    zero_division=0
))

print("\n=== TIME OF DAY RESULTS ===")
print("Accuracy:", accuracy_score(yt_true, yt_pred))
print(classification_report(
    yt_true,
    yt_pred,
    target_names=time_enc.classes_,
    zero_division=0
))
