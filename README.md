# Bank-Call-Center-Data-Analysis
This project analyzes telemarketing data from a bank to predict whether a customer will subscribe to a term deposit. The goal is to reduce wasted call costs (approx. €8 per call) by identifying customers who are likely to convert.

---

## 📁 Dataset

- **Source**: `bank-additional-full.csv`
- Models Used: Logistic Regression, Decision Tree
- **Attributes**: Customer demographic info, previous marketing interactions, economic indicators, and the subscription status (`y`).

---

## 📊 Objective

Build and evaluate machine learning models to classify whether a client will subscribe to a term deposit based on campaign and customer data.

---

## 🔍 Key Tasks

### 1. Data Loading & Exploration
- Load dataset
- Display sample rows
- Check class imbalance
- Visualize categorical impacts (job, education)
- Correlation analysis

### 2. Preprocessing
- Handle `unknown` as missing values
- Drop or encode missing values
- Apply `OneHotEncoding` for categorical variables
- Feature scaling using `StandardScaler`

### 3. Feature Importance
- Used `RandomForestClassifier` to determine top influential features
- Visualized top 10 features with bar chart

### 4. Model Training
- Train/Test Split: 80% / 20%
- Models used:
  - **Logistic Regression**
  - **Decision Tree (with GridSearchCV)**

### 5. Model Evaluation
- Metrics:
  - Classification report
  - ROC AUC Score
- Compared models based on ROC AUC

---

## ✅ Results

| Model              | ROC AUC |
|-------------------|---------|
| Logistic Regression | `~0.XX` |
| Decision Tree       | `~0.XX` |

Final decision is based on ROC AUC. The model with the higher score is considered better.

---

## 📈 Visual Outputs

- `subscription_distribution.png`: Target class balance
- `correlation_heatmap.png`: Correlation between numeric features
- `job_vs_subscription.png`, `education_vs_subscription.png`: Impact of categorical variables
- `feature_importance.png`: Top 10 features

---

## 🛠 Requirements

```bash
pip install pandas numpy seaborn matplotlib scikit-learn

🧠 Conclusion
This ML model can help the bank reduce cost by targeting customers more likely to subscribe. It also shows how classical ML techniques like Logistic Regression and Decision Tree can be effective with proper preprocessing and tuning.

