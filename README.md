# 🩺 Diabetes Prediction Web App

This is a **Diabetes Prediction Web Application** built using **Python Django** for the backend and **HTML, CSS, JavaScript** for the frontend. It utilizes a machine learning model trained on a publicly available **Diabetes Dataset from Kaggle** to predict whether a person is likely to have diabetes or not based on medical input features.

---

## 🚀 Features

- User-friendly web interface
- Predicts diabetes using medical parameters
- Trained ML model using Kaggle dataset
- Responsive frontend using HTML, CSS, JavaScript
- Integrated with Django backend
- Real-time form input and prediction

---

## 📂 Tech Stack

| Area      | Technology Used               |
|-----------|-------------------------------|
| Frontend  | HTML, CSS, JavaScript         |
| Backend   | Python Django                 |
| ML Model  | Scikit-learn                  |
| Dataset   | Kaggle - Pima Indian Diabetes Dataset |
| Others    | Django Templates, Bootstrap (optional)

---

## 📊 Dataset

- **Source:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features used:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

---

## 🧠 Machine Learning

- Data Preprocessing (Handling missing values, normalization)
- Model used: SVM
- Accuracy: 78%
- Model saved using `pickle` and integrated with Django views

---

## 🛠️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
