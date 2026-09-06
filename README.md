# **Fraud Detection in Financial Transactions**

## **Overview**
Financial fraud detection is a critical challenge for banks and financial institutions. This project leverages machine learning techniques to detect fraudulent credit card transactions using the Kaggle **Credit Card Fraud Detection Dataset**. The goal is to develop an accurate and reliable fraud detection model that minimizes false negatives while maintaining a low false positive rate.

---

## **Dataset**
### **Source:**
The dataset is sourced from Kaggle and contains anonymized features derived from PCA transformation. The key attributes include:
- **Time**: Time elapsed since the first transaction.
- **V1 - V28**: Principal Component Analysis (PCA) transformed features.
- **Amount**: Transaction amount.
- **Class**: The target variable (0 = Legitimate, 1 = Fraudulent).

### **Dataset Acknowledgement:**
[Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## **Objective**
- Detect fraudulent transactions using machine learning models.
- Reduce false negatives (fraud cases predicted as legitimate).
- Compare multiple models, fine-tune hyperparameters, and optimize decision thresholds.
- Use precision-recall tradeoff to balance fraud detection performance.

---

## **Methodology**
### **1. Exploratory Data Analysis (EDA)**
- Checked dataset information and missing values.
- Analyzed the distribution of fraud vs. legitimate transactions.
- Examined feature correlation using a heatmap.
- Identified important features using feature importance scores.

### **2. Data Preprocessing**
- **Handled Outliers:** Applied capping technique for features V14 and V17.
- **Log Transformation:** Applied log transformation on Amount to handle skewness.
- **Standardization:** Used StandardScaler for feature scaling.
- **Class Imbalance Handling:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance fraud and legitimate transactions.

### **3. Model Training & Evaluation**
The following models were trained and compared:
#### **Baseline Model:**
- **Logistic Regression**
  - ROC-AUC Score: **0.92**
  - Recall (Fraud Class): **85%**

#### **Tree-Based Models:**
- **Random Forest**
  - ROC-AUC Score: **0.96** (Full feature set)
  - Threshold Tuned (0.4): **Precision: 83.6%, Recall: 79.7%**
- **XGBoost**
  - ROC-AUC Score: **0.97**
  - Threshold Tuned (0.7): **Precision: 37.5%, Recall: 82.4%**

### **4. Threshold Tuning for Decision Optimization**
- Fine-tuned thresholds for Random Forest & XGBoost.
- Selected **Threshold = 0.4** for **Random Forest** and **0.7** for **XGBoost** based on Precision-Recall tradeoff.

### **5. Final Model Selection**
- **Random Forest with Threshold 0.4** was chosen as the final model due to a better balance between precision and recall.
- **Final ROC-AUC Score: 0.96**

---

## **Model Deployment Preparation**
### **Saved Artifacts:**
- **Final Model:** `final_rf_model.pkl`
- **Scaler:** `scaler.pkl`
- **Decision Threshold:** `threshold.pkl`

### **Usage:**
1. **Load the Model & Scaler**:
```python
import joblib
rf_model = joblib.load("final_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold_value = joblib.load("threshold.pkl")
```

2. **Make Predictions on New Transactions**:
```python
new_data_scaled = scaler.transform(new_data)
probabilities = rf_model.predict_proba(new_data_scaled)[:, 1]
predictions = (probabilities >= threshold_value).astype(int)
```

---

## **Conclusion**
This project successfully implemented machine learning techniques for fraud detection. By applying **Random Forest with threshold tuning**, we improved the recall of fraudulent transactions while maintaining a high precision rate. Future improvements can include deep learning models or real-time fraud detection system integration.

---

## **Future Work**
- Implement **Anomaly Detection** techniques.
- Try **Deep Learning (LSTMs, Autoencoders) for sequential fraud detection**.
- Deploy the model as an **API for real-time fraud detection**.

---

## **Repository Structure**
```
├── creditcard.csv/           # Dataset files
├── FraudDetection.ipybn/     # Jupyter Notebooks
├── README.md                 # Project documentation
└── requirements.txt          # Required dependencies
```

---

## **Installation & Usage**
### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/Fraud-Detection-ML.git
cd Fraud-Detection-ML
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run Jupyter Notebook**
```bash
jupyter notebook
```

---

## **Acknowledgments**
- Kaggle for providing the **Credit Card Fraud Detection Dataset**.
- Scikit-learn, XGBoost, and Pandas for making machine learning accessible.

---

🚀 **Developed by Aman Jain**
