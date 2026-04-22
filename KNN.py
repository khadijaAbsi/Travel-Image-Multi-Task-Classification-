import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# CONFIG

IMAGE_FOLDER = "travel_images"
IMG_SIZE = 64  # لتسريع KNN
TEST_SIZE = 0.2
RANDOM_STATE = 42
K = 1  # الأفضل مع داتاسيتك


# LOAD & PREPROCESS DATA

images = []
weather_labels = []
time_labels = []

for file in os.listdir(IMAGE_FOLDER):
    if file.endswith((".jpg", ".png", ".jpeg")):
        try:
            weather, time, _ = file.split("_", 2)
            img = cv2.imread(os.path.join(IMAGE_FOLDER, file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # رمادي لتسريع
            img = img.flatten() / 255.0  # flatten الصورة

            images.append(img)
            weather_labels.append(weather)
            time_labels.append(time)
        except:
            pass

X = np.array(images)


# ENCODE LABELS

weather_enc = LabelEncoder()
time_enc = LabelEncoder()

y_weather = weather_enc.fit_transform(weather_labels)
y_time = time_enc.fit_transform(time_labels)


# SPLIT DATA

X_train, X_test, yw_train, yw_test, yt_train, yt_test = train_test_split(
    X, y_weather, y_time, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# TRAIN MULTI-TASK KNN

knn_weather = KNeighborsClassifier(n_neighbors=K)
knn_time = KNeighborsClassifier(n_neighbors=K)

knn_weather.fit(X_train, yw_train)
knn_time.fit(X_train, yt_train)


# PREDICT BOTH TASKS

yw_pred = knn_weather.predict(X_test)
yt_pred = knn_time.predict(X_test)


# EVALUATE

print("=== WEATHER RESULTS (K=1) ===")
print("Accuracy:", accuracy_score(yw_test, yw_pred))
print(classification_report(yw_test, yw_pred, target_names=weather_enc.classes_, zero_division=0))

print("\n=== TIME OF DAY RESULTS (K=1) ===")
print("Accuracy:", accuracy_score(yt_test, yt_pred))
print(classification_report(yt_test, yt_pred, target_names=time_enc.classes_, zero_division=0))
