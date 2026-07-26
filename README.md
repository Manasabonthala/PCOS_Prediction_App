# 🩺 PCOS Prediction Using Deep Learning

A Streamlit-based web application that predicts the likelihood of **Polycystic Ovary Syndrome (PCOS)** using an Artificial Neural Network (ANN) trained on clinical patient data.

---

## 📌 Overview

Polycystic Ovary Syndrome (PCOS) is one of the most common hormonal disorders affecting women of reproductive age. Early prediction can help in timely diagnosis and lifestyle management.

This project uses a Deep Learning model to predict whether a patient is at **High Risk** or **Low Risk** of PCOS based on clinical parameters and symptoms.

---

## ✨ Features

- 🧠 Deep Learning (Artificial Neural Network)
- 📊 PCOS Risk Prediction
- 📈 Prediction Probability
- 📋 Patient Summary
- 💡 Personalized Health Recommendations
- 🩸 Blood Group Encoding
- 🌐 Interactive Streamlit Web Application

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```text
PCOS_Prediction_App/
│
├── app.py
├── pcos_ann_model.keras
├── scaler.pkl
├── requirements.txt
├── README.md
├── PCOS_Prediction_ANN.ipynb
│
└── images/
    ├── high_risk.png
    └── low_risk.png
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PCOS_Prediction_App.git
cd PCOS_Prediction_App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📷 Application Screenshots

### 🔴 High Risk Prediction

![High Risk Prediction](images/high_risk.png)

---

### 🟢 Low Risk Prediction

![Low Risk Prediction](images/low_risk.png)

---

## 🧠 Model Information

- Model: Artificial Neural Network (ANN)
- Framework: TensorFlow / Keras
- Input Features: 48 Clinical Parameters
- Output:
  - High Risk of PCOS
  - Low Risk of PCOS

---

## 📋 Input Features

The model uses the following patient information:

- Age
- Weight
- Height
- BMI
- Pulse Rate
- Respiratory Rate
- Hemoglobin
- Menstrual Cycle Information
- Pregnancy Details
- Hormonal Parameters (FSH, LH, AMH, TSH, PRL)
- Blood Sugar
- Blood Pressure
- Ultrasound Measurements
- Symptoms
- Blood Group

---

## 🎯 Future Improvements

- Explainable AI (XAI)
- Model Confidence Visualization
- PDF Report Generation
- User Authentication
- Cloud Deployment
- Doctor Dashboard

---

## 👩‍💻 Author

** Bonthala Manasa **

Computer Science Engineering Student

Deep Learning • Machine Learning • Data Science • Python

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub!