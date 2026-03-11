# 💳 Predicting Credit Card Spend & Identifying Key Drivers

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Regression](https://img.shields.io/badge/ML-Linear%20Regression%20%7C%20Decile%20Analysis-orange?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Banking%20%7C%20FinTech-lightblue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-5000%20Customers-green?style=flat-square)
![Stars](https://img.shields.io/github/stars/vicky60629/Predicting-Credit-Card-Spend?style=flat-square)

> **A regression-based ML project that identifies the key drivers of credit card spending across 5,000 bank customers — enabling data-driven credit limit calculation for new applicants.**

---

## 📌 Problem Statement

A global bank wants to understand **what factors drive credit card spend** across its customer base. This understanding is critical for:
- Setting accurate **credit limits** for new applicants
- Identifying high-value customer segments
- Reducing credit risk by predicting spend behaviour upfront

The bank surveyed **5,000 customers**, collecting demographic, financial, and behavioural data. This project builds a predictive model to:
1. Identify the **top drivers** of total credit card spend (Primary + Secondary card combined)
2. Predict the **total spend** for new applicants given their profile
3. Use those predictions to **recommend credit limits** at the individual level

---

## 🎯 Business Impact

| Stakeholder | Value Delivered |
|-------------|----------------|
| **Credit Risk Team** | Data-driven credit limit setting, reducing over/under-lending |
| **Marketing Team** | Identify high-spend customer profiles to target premium products |
| **Operations** | Predict customer lifetime value early in the relationship |
| **Product Team** | Understand which customer segments drive the most revenue |

---

## 📊 Dataset

- **Source:** Internal bank customer survey
- **Size:** 5,000 customers
- **Format:** `.xlsx` with accompanying data dictionary
- **Target Variable:** `total_spent` (Primary Card spend + Secondary Card spend, log-transformed)

### Key Features

| Category | Variables |
|----------|-----------|
| **Demographics** | Age (`agecat`), Education (`edcat`), Marital status (`spoused`) |
| **Financial** | Income category (`inccat`), Credit limit, Minimum payments |
| **Behavioural** | Tenure (`tenure`, `card2tenure`), Equipment months (`equipmon`) |
| **Geographic** | Address duration (`addresscat`), Commute category (`commutecat`) |
| **Communication** | Toll months (`tollmon`), Wireless months (`wiremon`) |

> **Note:** Data is numerically encoded. Categorical variables require careful separation before modelling.

---

## 🏗️ Project Architecture

```
Predicting-Credit-Card-Spend/
├── data/
│   ├── credit_card_data.xlsx        # Raw customer survey data
│   └── Data_Dictionary.xlsx         # Variable definitions
├── outputs/
│   ├── Decile_analysis_train.csv    # Decile analysis on training set
│   ├── Decile_analysis_test.csv     # Decile analysis on test set
│   └── python_FA.xls                # Factor analysis output
├── Predicting Credit Card Spend.ipynb   # Full EDA + Modelling notebook
├── requirements.txt
└── README.md
```

---

## 🔍 Approach

### 1. Data Preparation & Cleaning

**Key challenges tackled:**
- Numerical encoding masks categorical variables — required careful type separation
- Missing value treatment: variables with >25% missing dropped; remainder imputed
- Outlier treatment using **1st–99th percentile clipping** to reduce skew

```python
# Outlier clipping at 1st and 99th percentile
credit[numerical_var] = credit[numerical_var].apply(
    lambda x: x.clip(lower=x.dropna().quantile(0.01),
                     upper=x.quantile(0.99))
)
```

**Variables dropped due to multicollinearity:**
```python
drop_col = ['addresscat', 'agecat', 'cardtenurecat', 'card2tenurecat',
            'cardtenure', 'commutecat', 'edcat', 'equipmon', 'inccat',
            'lnlongten', 'longten', 'spoused', 'spousedcat', 'tenure',
            'tollmon', 'wiremon']
```

### 2. Feature Engineering

- **Log transformation** on target variable (`total_spent_ln`) to normalize right-skewed spend distribution
- **Dummy variable creation** for categorical variables (excluding boolean variables which are already binary)
- **Missing value analysis** — calculated % missing per variable to guide imputation strategy

```python
Missing_values = pd.DataFrame(
    credit_conti_var.apply(continuous_var_summary).T.round(1)['NMISS'] / 5000 * 100
)
# Drop variables with >25% missing values
M_V = Missing_values[Missing_values['NMISS'] > 25]
```

### 3. Modelling

- **Linear Regression** as the primary model — interpretable, directly maps to business insight
- Train/test split for out-of-sample validation
- Focus on **coefficient interpretation** to identify key spend drivers

### 4. Decile Analysis (Business Validation)

A key differentiator of this project — **Decile Analysis** validates model performance in business terms:

```python
# Group predictions into deciles and compare predicted vs actual spend
Predicted_avg = train[['Deciles', 'pred_tot_spend']].groupby(
    train.Deciles).mean().sort_index(ascending=False)['pred_tot_spend']

Actual_avg = train[['Deciles', 'total_spent_ln']].groupby(
    train.Deciles).mean().sort_index(ascending=False)['total_spent_ln']
```

This shows whether the model correctly ranks customers from **highest to lowest spenders** — critical for credit limit assignment.

---

## 📈 Key Findings

- **Income category** is the single strongest predictor of total spend — higher income customers spend significantly more
- **Card tenure** (how long the customer has held the card) positively correlates with spend — loyal customers spend more
- **Secondary card usage** is a strong signal — customers using both primary and secondary cards show 2x higher total spend
- Customers with **longer address tenure** tend to be more financially stable with predictable spend patterns
- **Log transformation** of the target variable significantly improved model fit by reducing skew

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Modelling | Scikit-learn (Linear Regression) |
| Statistical Analysis | SciPy, StatsModels |
| Output | Excel (openpyxl / xlwt) for decile reports |
| Notebook | Jupyter Notebook |

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/vicky60629/Predicting-Credit-Card-Spend.git
cd Predicting-Credit-Card-Spend

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook "Predicting Credit Card Spend.ipynb"
```

---

## 📊 Decile Analysis Output

The model's business performance is validated through **decile-level comparison** of predicted vs actual spend — a standard technique used in banking and insurance to evaluate regression models.

| Decile | Avg Predicted Spend | Avg Actual Spend |
|--------|--------------------:|----------------:|
| 1 (Top) | Highest | Highest |
| 2 | ↓ | ↓ |
| ... | ... | ... |
| 10 (Bottom) | Lowest | Lowest |

A well-performing model shows **monotonically decreasing** averages from Decile 1 to 10, confirming it correctly ranks customers by spend potential.

---

## 💡 Key Learnings & Future Improvements

**What worked well:**
- Decile analysis goes beyond standard RMSE/R² — validates the model in business language that stakeholders understand
- Careful missing value and outlier treatment significantly improved model stability
- Log transformation of spend target made residuals more normally distributed

**Future enhancements:**
- [ ] Try **XGBoost / LightGBM** for potentially better predictive accuracy
- [ ] Add **SHAP values** for better feature importance explainability
- [ ] Build a **Streamlit dashboard** for interactive credit limit prediction
- [ ] Experiment with **Ridge / Lasso regression** for automatic feature selection
- [ ] Extend to **credit limit recommendation engine** using predicted spend buckets
- [ ] Track experiments with **MLflow**

---

## 👨‍💻 About the Author

**Vicky Gupta** — Data Engineering Analyst @ Accenture (4.5 years) | Aspiring Data Scientist

Experienced in PySpark, ETL pipelines, and building end-to-end ML solutions across banking, retail, and NLP domains.

🔗 [LinkedIn](https://www.linkedin.com/in/vicky-gupta-583016168/) | [GitHub](https://github.com/vicky60629)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this useful, please consider starring the repository!**
