# ------------------
# IMPORT LIBRARIES
# ------------------
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ------------------
# CONFIG
# ------------------
IMAGE_FOLDER = "travel_images"
IMG_SIZE = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ------------------
# LOAD & PREPROCESS DATA
# ------------------
images = []
weather_labels = []
time_labels = []

for file in os.listdir(IMAGE_FOLDER):
    if file.endswith((".jpg", ".png", ".jpeg", ".jfif", ".avif", ".webp")):
        try:
            weather, time, _ = file.split("_", 2)
            img = cv2.imread(os.path.join(IMAGE_FOLDER, file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # normalization
            images.append(img)
            weather_labels.append(weather)
            time_labels.append(time)
        except:
            pass

X = np.array(images)

# ------------------
# ENCODE LABELS
# ------------------
weather_enc = LabelEncoder()
time_enc = LabelEncoder()

y_weather = weather_enc.fit_transform(weather_labels)
y_time = time_enc.fit_transform(time_labels)

# طباعة الترتيب لمعرفة الأرقام
print("=== Weather Classes Mapping ===")
for i, label in enumerate(weather_enc.classes_):
    print(f"{i}: {label}")

print("\n=== Time Classes Mapping ===")
for i, label in enumerate(time_enc.classes_):
    print(f"{i}: {label}")

# DataFrame لسهولة التحليل
df = pd.DataFrame({
    "weather": weather_labels,
    "time": time_labels
})

# ------------------
# TRAIN/TEST SPLIT
# ------------------
X_train, X_test, yw_train, yw_test, yt_train, yt_test = train_test_split(
    X, y_weather, y_time, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ------------------
# EDA: CLASS DISTRIBUTION
# ------------------
plt.figure(figsize=(6,4))
sns.countplot(x="weather", data=df)
plt.title("Weather Class Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="time", data=df)
plt.title("Time of Day Class Distribution")
plt.show()

# ------------------
# EDA: SUMMARY STATISTICS
# ------------------
print("\n=== Weather Counts ===")
print(df["weather"].value_counts())
print("\n=== Time of Day Counts ===")
print(df["time"].value_counts())

# ------------------
# DEFINE CUSTOM COLORS
# ------------------

# Weather Colors (حسب الترتيب الأبجدي من LabelEncoder)
# Cloudy(0), NotClear(1), Rainy(2), Snowy(3), Sunny(4)
weather_color_map = {
    0: '#9C27B0',  # Purple - Cloudy
    1: '#2196F3',  # Blue - NotClear
    2: '#00BCD4',  # Cyan - Rainy
    3: '#4CAF50',  # Green - Snowy
    4: '#FFEB3B'   # Yellow - Sunny
}

# Time Colors (حسب الترتيب الأبجدي من LabelEncoder)
# Afternoon(0), Evening(1), Morning(2)
time_color_map = {
    0: '#E91E63',  # Pink - Afternoon
    1: '#FFEB3B',  # Yellow - Evening
    2: '#0D47A1'   # Dark Blue - Morning
}

# تحويل الأرقام لألوان
weather_colors = [weather_color_map[w] for w in y_weather]
time_colors = [time_color_map[t] for t in y_time]

# ------------------
# EDA: PCA SCATTER PLOTS WITH CUSTOM COLORS
# ------------------
X_flat = X.reshape(len(X), -1)  # flatten الصور
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)

# ========================================
# WEATHER PCA PLOT
# ========================================
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=weather_colors, alpha=0.7, s=30)

plt.title("PCA Projection of Weather Classes", fontsize=14, fontweight='bold')
plt.xlabel("PC1", fontsize=12)
plt.ylabel("PC2", fontsize=12)

# إضافة Legend يدوي
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#9C27B0', label='Cloudy (0)'),
    Patch(facecolor='#2196F3', label='NotClear (1)'),
    Patch(facecolor='#00BCD4', label='Rainy (2)'),
    Patch(facecolor='#4CAF50', label='Snowy (3)'),
    Patch(facecolor='#FFEB3B', label='Sunny (4)')
]
plt.legend(handles=legend_elements, loc='best', title='Weather Class')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ========================================
# TIME PCA PLOT
# ========================================
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=time_colors, alpha=0.7, s=30)

plt.title("PCA Projection of Time of Day Classes", fontsize=14, fontweight='bold')
plt.xlabel("PC1", fontsize=12)
plt.ylabel("PC2", fontsize=12)

# إضافة Legend يدوي
legend_elements = [
    Patch(facecolor='#0D47A1', label='Morning (2)'),
    Patch(facecolor='#E91E63', label='Afternoon (0)'),
    Patch(facecolor='#FFEB3B', label='Evening (1)')
]
plt.legend(handles=legend_elements, loc='best', title='Time of Day Class')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
