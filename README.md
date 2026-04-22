# 🌍 Travel Image Multi-Task Classification

This project focuses on **multi-task image classification** using travel images to predict:

* 🌤️ **Weather Condition** (Cloudy, NotClear, Rainy, Snowy, Sunny)
* 🌅 **Time of Day** (Morning, Afternoon, Evening)

The system applies multiple machine learning approaches and compares their performance.

---

## 📌 Project Overview

This project was developed as part of the **Machine Learning and Data Science (ENCS5341)** course.

We tackled a **dual classification problem** using the same input image:

* Predicting environmental conditions (weather)
* Predicting temporal context (time of day)

---

## 🧠 Models Implemented

We implemented and compared three different approaches:

### 🔹 1. K-Nearest Neighbors (KNN)

* Simple baseline model
* Works on flattened grayscale images
* Best configuration: `K = 1`

---

### 🔹 2. Support Vector Machine (SVM)

* Kernel: RBF
* Handles non-linear decision boundaries
* Uses `MultiOutputClassifier` for multi-task prediction
* Best configuration: `C = 10`

---

### 🔹 3. Convolutional Neural Network (CNN)

* Based on **MobileNetV2 (Transfer Learning)**
* Multi-output architecture:

  * Weather head
  * Time-of-day head
* Shared feature extractor

---

## 📊 Dataset

* Total images: **904**
* Weather classes: **5**
* Time classes: **3**

### ⚠️ Challenges:

* Severe class imbalance:

  * Sunny: 55%
  * Rainy: only 2.7%
* Visual similarity between classes
* Weak separability (confirmed using PCA)

---

## 🔍 Data Processing

* Image resizing (64×64 or 224×224)
* Normalization (pixel scaling)
* Label extraction from filenames:

  ```
  Sunny_Morning_001.jpg
  ```
* Encoding using `LabelEncoder`

---

## 📈 Exploratory Data Analysis (EDA)

* Class distribution visualization (Seaborn)
* PCA projection for feature space visualization
* Observations:

  * Strong overlap in weather classes
  * Better separation for time-of-day (especially Evening)

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Macro & Weighted averages

---

## 📊 Results Summary

| Model      | Weather Accuracy | Time Accuracy | F1 Score |
| ---------- | ---------------- | ------------- | -------- |
| KNN (K=1)  | ~59%             | ~64%          | ~0.54    |
| SVM (C=10) | ~59%             | ~63%          | ~0.58    |
| CNN        | ~60%             | ~61%          | ~0.57    |

---

## 🏆 Key Findings

* All models reached a performance ceiling (~60%)
* SVM provided the best **cost-performance balance**
* CNN did not significantly outperform simpler models
* Class imbalance severely impacted performance
* Rainy class was almost impossible to learn

---

## ⚠️ Limitations

* Small dataset size
* Severe class imbalance (up to 14:1 ratio)
* Label noise from filename-based annotation
* High feature overlap between classes

---

## 🚀 Future Improvements

* Collect more balanced data
* Apply smarter augmentation
* Use attention-based models
* Perform sky segmentation carefully
* Try ensemble methods

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy / Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib / Seaborn

---

## 👩‍💻 Authors

* Khadija Absi
* Raghad Qadus
* Maisam AbuJaber

---

## 📚 Course Information

**Machine Learning and Data Science (ENCS5341)**
Faculty of Engineering & Technology
Instructor: Dr. Yazan Abu Farha

---

## 📌 Conclusion

The main limitation of this project is **data quality, not model complexity**.
Even advanced models could not surpass the performance ceiling due to:

* class imbalance
* insufficient minority samples

SVM emerged as the most practical solution.

---

⭐ If you find this project useful, feel free to star the repository!
