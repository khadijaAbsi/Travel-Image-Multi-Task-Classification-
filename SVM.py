import os
import cv2
import numpy as np
import warnings  # <--- to hide warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score


# HIDE WARNINGS

warnings.filterwarnings("ignore")  # <--- all warnings hidden


# CONFIG

IMAGE_FOLDER = "travel_images"
IMG_SIZE = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42


# LOAD & PREPROCESS DATA

images = []
weather_labels = []
time_labels = []

for file in os.listdir(IMAGE_FOLDER):
    if file.endswith((".jpg", ".png", ".jpeg", ".jfif", ".avif", ".webp")):
        try:
            weather, time, _ = file.split("_", 2)
            img = cv2.imread(os.path.join(IMAGE_FOLDER, file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # normalize pixel values
            images.append(img)
            weather_labels.append(weather)
            time_labels.append(time)
        except:
            pass

X = np.array(images)
X_flat = X.reshape(len(X), -1)  # flatten images for SVM


# ENCODE LABELS

weather_enc = LabelEncoder()
time_enc = LabelEncoder()

y_weather = weather_enc.fit_transform(weather_labels)
y_time = time_enc.fit_transform(time_labels)


# TRAIN/TEST SPLIT

X_train, X_test, yw_train, yw_test, yt_train, yt_test = train_test_split(
    X_flat, y_weather, y_time, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# DEFINE SVM WITH CLASS WEIGHTS

svm = SVC(
    kernel='rbf',
    C=100,
    gamma='scale',
    class_weight='balanced',
    probability=False
)

multi_svm = MultiOutputClassifier(svm)


# TRAIN MULTI-OUTPUT SVM

multi_svm.fit(X_train, np.vstack([yw_train, yt_train]).T)


# PREDICTION

y_pred = multi_svm.predict(X_test)
yw_pred = y_pred[:,0]
yt_pred = y_pred[:,1]


# EVALUATE

print("\n=== WEATHER RESULTS ===")
print("Accuracy:", accuracy_score(yw_test, yw_pred))
print(classification_report(yw_test, yw_pred, target_names=weather_enc.classes_))

print("\n=== TIME OF DAY RESULTS ===")
print("Accuracy:", accuracy_score(yt_test, yt_pred))
print(classification_report(yt_test, yt_pred, target_names=time_enc.classes_))
