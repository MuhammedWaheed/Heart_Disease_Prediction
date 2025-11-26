# 🤖 Test App From this link 
()

# ❤️ Heart Disease Prediction App

This project is an interactive **machine learning web application** built with **Streamlit** that predicts a user’s risk of heart disease based on their health information.  
Users simply enter their data, and the app returns a probability score along with a clean, animated result card.

---

### **📌 Dataset Size**

| Attribute |        Value           |
| --------- |      --------------- |
| Records   |      **50,000**      |
| Features  |      **20 + Target** |
| Format    |      **CSV**         |

---

## 🧩 Project Structure

```
Heart_Disease_Prediction/
│
├── App.py                 → Streamlit front‑end app
├── model.pkl              → Trained ML model (pipeline)
├── Heart_Disease.ipynb    → Model training notebook
├── README.md
└── requirements.txt
```

---

## 🛠️ How it Works

### 1️⃣ User enters health information  
- Age, gender, BMI, lifestyle  
- Blood pressure, cholesterol  
- Chronic conditions  
- Family history  

### 2️⃣ Model processes data through a pipeline  
- Missing value imputation  
- Scaling / encoding  
- Logistic Regression classifier  

### 3️⃣ App displays:
- Predicted class (High Risk / Low Risk)  
---

-----

## 🧠 Model Details

- Algorithm: Logistic Regression (GridSearch‑tuned)
- Preprocessing:
  - OneHotEncoding  
  - StandardScaler  
  - ColumnTransformer pipeline  
- Output:
  - Class prediction (0 = Low Risk, 1 = High Risk)
  - Probability score  

---

## 📘 Dataset

This project uses a curated heart disease dataset containing:  
- Demographics  
- Lifestyle factors  
- Clinical measurements  
- Medical history  
---

## ⚠️ Disclaimer

This app is **not** a medical diagnostic tool.  
Predictions are for **educational and preventive purposes only.**  
Always consult medical professionals for real decisions.

---
**Muhammed** — Data Scientist & Machine Learning Practitioner  
