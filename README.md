# Titanic-Survival
Predicts Titanic passenger survival using various classification models to find out the best ones, and tuning them right..

# 🚢 Titanic Survival Prediction

## 📌 Overview
This project predicts whether a passenger survived the Titanic disaster using **machine learning models**.  
It covers **data preprocessing, feature engineering, model comparison, hyperparameter tuning**, and **visualizations**.

---

## 📊 Dataset
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
- **Files Used:**  
  - `train.csv` → training data  
  - `test.csv` → test data for prediction  

Key preprocessing steps:  
- Filled missing `Age` with median values  
- Dropped `Cabin` due to too many missing values  
- Filled missing `Embarked` with mode  
- Encoded categorical features (`Sex`, `Embarked`) with LabelEncoder  
- Scaled numerical features with `StandardScaler`  

---

## ⚙️ Models Implemented
- Logistic Regression  
- Support Vector Classifier (SVC)  
- Gradient Boosting Classifier  
- Decision Tree  
- Random Forest  

### 🔍 Model Comparison
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~80.4% |
| SVC | ~81.5% |
| Gradient Boosting | ~79.8% |
| Decision Tree | ~78.2% |
| **Random Forest** | **~82.1%** |

---

## 🏆 Final Model
- Tuned with **RandomizedSearchCV**  
- Best parameters:  
  ```python
  {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 103}

📈 Visualizations

Countplot of predicted survival distribution
Pie chart showing proportion of predicted survivors vs non-survivors



