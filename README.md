# Insurance Fraud Detection using Machine Learning

---

## 1. Project Overview

### Problem Statement
Insurance companies lose millions due to fraudulent claims. Fraud detection is difficult because fraudulent cases are rare and hidden among legitimate claims.

### Objective
Build a machine learning classifier to detect whether an insurance claim is:

- Fraudulent  
- Legitimate  

### Key Metric
- **Recall for Fraud Class (Priority)**

### Business Value
- Reduce fraudulent payout risk  
- Assist fraud investigation teams  
- Improve decision-making efficiency  

---

## 2. Dataset Description

### Dataset Overview

- Total samples: **1000 insurance claims**
- Fraud cases: **247**
- Fraud percentage: **24.7%**
- Total features: Multiple numerical & categorical features
- Data types: Mixed (Categorical + Numerical)
- Imbalance observed: **Yes (Moderate)**
- Fraud class treated as **priority**

---

## 3. Data Preprocessing Steps

### Pipeline Overview

Raw Data
↓
Cleaning
↓
Missing Values
↓
Outlier Treatment
↓
EDA
↓
Feature Engineering
↓
Model Training


---

### Data Preparation

#### 1. Removal of Irrelevant Columns
- Dropped identifier-based columns (policy IDs, location IDs)
- Prevents noise and data leakage

---

#### 2. Data Type Correction
Converted:
- `policy_bind_date`
- `incident_date`

To:


---

#### 3. Missing Value Verification
- Checked using Pandas
- **No missing values found**

---

### Outcome
- Clean dataset  
- Correct data types  
- No missing values  
- No irrelevant features  

---

### Outlier Detection

- Used **IQR method**
- Feature analyzed: `policy_annual_premium`

| Bound | Value |
|------|------|
| Lower Bound | 600.48 |
| Upper Bound | 1904.83 |

- **9 outliers detected**

---

### Outlier Treatment
- Applied **log transformation (`log1p`)**
- Reduced skewness without removing data

### before 
<img width="750" height="580" alt="image" src="https://github.com/user-attachments/assets/bb0ea7cc-d4d6-468f-b5d7-bf67a9a2cb69" />
<img width="745" height="549" alt="image" src="https://github.com/user-attachments/assets/57857e0d-3e52-4cde-829c-e09bd4fcc5da" />

### after
<img width="745" height="553" alt="image" src="https://github.com/user-attachments/assets/9997717d-d632-4d0a-8ffb-a999204128a5" />

### Log Transformation
<img width="1497" height="500" alt="image" src="https://github.com/user-attachments/assets/dc35897e-53ab-4040-91f6-747e9e2cebad" />


---

### Descriptive Statistical Analysis
- Used `df.describe()`
- Observed:
  - Mean
  - Standard deviation
  - Min / Max
  - Quartiles
- Confirmed:
  - 1000 records
  - No missing values

---

### Correlation Analysis

- Used heatmap to detect multicollinearity

#### Removed Features:
- `months_as_customer`
- `injury_claim`
- `vehicle_claim`
- `property_claim`

**Reason:** Reduce redundancy and improve generalization

##correlation matrix
<img width="929" height="679" alt="image" src="https://github.com/user-attachments/assets/47af8ab8-33b7-4de6-b01d-5ef76b153faf" />


---

### Encoding Categorical Variables

#### Encoding Pipeline

Binary → Mapping (Y/N → 1/0)
Ordinal → Manual Mapping
Nominal → OneHotEncoding


#### Encoding Strategy

| Feature Type | Encoding |
|-------------|----------|
| Binary | Mapping |
| Nominal | OneHot |
| Ordinal | Manual |

#### Ordinal Mapping Example

| Category | Encoding |
|----------|----------|
| Trivial Damage | 0 |
| Minor Damage | 1 |
| Major Damage | 2 |
| Total Loss | 3 |

---

## 4. Feature Scaling

| Algorithm | Scaling Needed |
|----------|---------------|
| Logistic Regression | Yes |
| KNN | Yes |
| SVM | Yes |
| Decision Tree | No |
| Random Forest | No |

### Workflow

Encoding
↓
Train-Test Split
↓
Fit scaler on X_train
↓
Transform X_train & X_test


---

## 5. Train-Test Split & Data Handling

### Pipeline

1. Split into X and y  
2. Train-test split  
3. Apply SMOTE on training data only  
4. Apply StandardScaler  

---

### SMOTE Explanation

- Balances minority class (fraud)
- Generates synthetic data points
- Applied only on training set to avoid data leakage

---

### Final Dataset

| Variable | Purpose |
|----------|--------|
| X_train_smote | Training (tree models) |
| X_test | Testing |
| X_train_scaled | Training (scaled models) |
| X_test_scaled | Testing |
| y_train_smote | Training labels |
| y_test | Test labels |

---

## 6. Model Training

### Handling Imbalanced Data

#### Approach A — Baseline
- No balancing
- Used for comparison

---

#### Approach B — SMOTE
- Synthetic fraud samples created
- Applied only to training data

---

#### Approach C — Class Weight
- Penalizes fraud misclassification more heavily

---

## 7. Model Comparison Table

### Approach Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|------|----------|----------|--------|----------|
| Baseline | 74.5% | 0.00 | 0.00 | 0.00 |
| SMOTE | 56.5% | 0.27 | 0.47 | 0.35 |
| Class Weight | 51.5% | 0.26 | 0.51 | 0.34 |

### Insight

- Baseline fails to detect fraud  
- SMOTE and Class Weight significantly improve recall  

---

### Detailed Model Comparison

| Model | Approach | Accuracy | Precision | Recall | F1 |
|------|---------|----------|----------|--------|-----|
| Decision Tree | Baseline | 0.775 | 0.5417 | 0.5306 | 0.5361 |
| Decision Tree | SMOTE | 0.770 | 0.5283 | 0.5714 | 0.5490 |
| Decision Tree | Class Weight | 0.785 | 0.5600 | 0.5714 | 0.5657 |
| Random Forest | Baseline | 0.745 | 0.3750 | 0.0612 | 0.1053 |
| Random Forest | SMOTE | 0.765 | 0.5385 | 0.2857 | 0.3733 |
| Random Forest | Class Weight | 0.750 | 0.4444 | 0.0816 | 0.1379 |
| KNN | Baseline | 0.665 | 0.1786 | 0.1020 | 0.1299 |
| KNN | SMOTE | 0.495 | 0.2347 | 0.4694 | 0.3129 |
| SVM | Baseline | 0.755 | 0.0000 | 0.0000 | 0.0000 |
| SVM | SMOTE | 0.650 | 0.2439 | 0.2041 | 0.2222 |
| Naive Bayes | Baseline | 0.750 | 0.4545 | 0.1020 | 0.1667 |
| Naive Bayes | SMOTE | 0.455 | 0.2857 | 0.8163 | 0.4233 |

---

## 8. Model Selection

### Final Model
- **Decision Tree with Class Weight**

### Reason
- Highest recall for fraud detection  
- Balanced F1-score  
- Stable performance  

---

### Hyperparameter Tuning

- Parameter used: `max_depth = 5`
- Accuracy improved to **~80%**

---

## 9. Evaluation Metrics

Used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### Key Insight

> Accuracy alone is misleading in imbalanced datasets  
> Recall is the most critical metric  

---

## 10. Business Interpretation

- Model prioritizes fraud detection  
- Minimizes financial loss  
- Improves investigation efficiency  

---

## 11. Deployment

### Objective Flow

User Input
↓
Preprocessing
↓
Feature Vector (143 features)
↓
Model.predict()
↓
Result


---

### System Design

#### Frontend
- HTML, CSS, JavaScript
- Input form

#### Backend
- Flask
- Handles preprocessing + prediction

#### Model Layer
- Saved `.pkl` Decision Tree model

---

## 12. How to Run

```bash
git clone <repo-link>
cd insurance-fraud-detection
pip install -r requirements.txt
python app/app.py
```

Open
```http://127.0.0.1:5000```

## 13. Project Structure

```bash
insurance-fraud-detection/
│
├── data/
├── notebooks/
├── src/
├── model/
├── app/
├── requirements.txt
├── README.md
└── report.pdf
```

