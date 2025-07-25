
#  Startup Scoring Model – ScaleDux AI Intern Task 1

Welcome to the **Startup Scoring Model** project!
This repository contains a full machine learning workflow to predict the **composite score of startups** based on key features like team experience, market size, and funding raised.
The goal is to evaluate different ML models and build an effective scoring system that ranks startups based on their potential.

**Notebook Link:** [Ritesh\_Ranjan\_ScaleDux\_AI\_Intern\_Task1.ipynb](https://github.com/Ritesh-GitHub-Ranjan/Scaledux_Assignment/blob/main/Ritesh_Ranjan_ScaleDux_AI_Intern_Task1.ipynb)

---

##  Project Structure

```
.
├── Ritesh_Ranjan_ScaleDux_AI_Intern_Task1.ipynb
├── Startup_Scoring_Dataset.csv
├── Output/Graphs
├── README.md   ← You're here!
```

---

##  Problem Statement

We are given a dataset of startups with the following features:

* `team_experience` – Experience level of the startup team
* `market_size_million_usd` – Size of the target market
* `monthly_active_users` – Number of active users
* `monthly_burn_rate_inr` – Monthly cost or spending
* `funds_raised_inr` – Investment received
* `valuation_inr` – Current valuation of the startup

The task is to **predict a composite score** representing startup quality, based on the above features.

---

##  Tools & Libraries Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn (for visualization)

---

##  Workflow Summary

### 1. **Data Preprocessing**

* Loaded the dataset
* Checked for nulls and outliers
* Normalized the `composite_score` using Min-Max Scaling for better model performance

### 2. **Model Building**

We implemented and compared the following models:

#### A. **Linear Regression**

* Simple baseline model
* Good for understanding linear relationships
* RMSE (Test): **13.59**

#### B. **XGBoost Regressor**

* More advanced model that handles non-linearity well
* RMSE (Test): **12.67**
* RMSE (CV): **13.68**

---

### 3. **Feature Engineering**

* Applied **log transformations** on skewed features like `market_size`, `burn_rate`, etc.
* Added **interaction term**:
  `experience_market_interaction = team_experience * market_size`

---

### 4. **Hyperparameter Tuning**

Used **GridSearchCV** to tune key XGBoost parameters:

```python
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 150]
}
```

**Best Parameters Found:**

```bash
{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}
```

---

### 5. **Model Stacking**

Combined Linear Regression + XGBoost using a **Stacking Regressor**:

```python
StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('xgb', best_xgb)
    ],
    final_estimator=Ridge()
)
```

However, this model **underperformed**, with RMSE (Test): **195.68**
➡ Suggests possible overfitting or data leakage.

---

### 6. **Model Comparison (Visual)**

A bar chart was created to compare model RMSEs across:

* Linear Regression
* XGBoost
* Cross-validation scores
* Stacked Model

---

##  Results Summary

| Model                  | RMSE (Test) |
| ---------------------- | ----------- |
| Linear Regression      | 13.59       |
| XGBoost                | 12.67       |
| Linear Regression (CV) | 13.11       |
| XGBoost (CV)           | 13.68       |
| Stacked Model          | ❌ 195.68    |

**Best Model:** XGBoost Regressor with tuned hyperparameters

---

## Suggest Improvements or Future Steps

Here are several ways to improve this project further:

### **Modeling Improvements**

* Fix the stacked model by using a better **meta-learner** (e.g., Lasso, Gradient Boosting)
* Use **robust scaling or standardization** before stacking
* Try **LightGBM** or **CatBoost** as alternatives to XGBoost

### **Feature Engineering**

* Create more interaction features (e.g., burn rate vs. valuation)
* Use **domain knowledge** to derive meaningful ratios (e.g., burn rate per user)
* Apply **clustering** to group similar startups and use cluster IDs as features

### ⚙ **Hyperparameter Optimization**

* Use **Optuna** or **Bayesian Optimization** instead of GridSearchCV for faster tuning
* Tune other parameters like `subsample`, `colsample_bytree`, etc., in XGBoost

###  **Evaluation & Validation**

* Use **Stratified K-Fold CV** if applicable
* Add **residual plots** to visualize model errors
* Use **MAE** or **R² score** alongside RMSE

###  **Deployment & Application**

* Wrap the model in a **Streamlit or Flask app** for demo
* Provide a scoring interface for real-time prediction
* Deploy it via **Render**, **Heroku**, or **Hugging Face Spaces**

---

## Contact

**Ritesh Ranjan**
Email: [riteshranjan1729@gmail.com](mailto:riteshranjan1729@gmail.com)
GitHub: [@Ritesh-GitHub-Ranjan](https://github.com/Ritesh-GitHub-Ranjan)


