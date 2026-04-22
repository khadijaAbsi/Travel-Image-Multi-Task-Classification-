# 🌍 Travel Image Multi-Task Classification

This project performs **multi-task image classification** to predict:

* 🌤️ Weather (Cloudy, NotClear, Rainy, Snowy, Sunny)
* 🌅 Time of Day (Morning, Afternoon, Evening)

---

## 🧠 Models Used

* **KNN** (baseline)
* **SVM** (best performance)
* **CNN (MobileNetV2)** (deep learning approach)

---

## 📊 Dataset

* 904 images
* 5 weather classes
* 3 time-of-day classes

⚠️ Imbalanced data (e.g., Rainy is very underrepresented)

---

## ⚙️ Features

* Image preprocessing & normalization
* Label extraction from filenames
* Multi-task learning (single model, multiple outputs)
* PCA visualization for data analysis

---

## 📈 Results

* Accuracy ~60% across all models
* SVM achieved best balance between performance and efficiency
* Performance limited by dataset quality and imbalance

---

## ⚠️ Limitations

* Small dataset
* Class imbalance
* Overlapping features between classes

---

## 🚀 Future Work

* Collect more balanced data
* Apply augmentation
* Try advanced models

---

## 🛠️ Tech Stack

Python, OpenCV, NumPy, Scikit-learn, TensorFlow, Matplotlib
