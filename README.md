# 📊 Machine Learning Test - Quick Reference

## 📁 Files Overview

### Q1.ipynb - Basic Statistics 📈

Calculate central tendency and spread measures for a dataset.

**Key Concepts:**

- 📍 **Mean** - Average of all values
- 🎯 **Median** - Middle value when sorted
- 🔢 **Mode** - Most frequent value
- 📏 **Range** - Difference between max and min

**Formula to Remember:**

```python
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data).mode  # No [0] indexing!
range = np.max(data) - np.min(data)
```

---

### Q2.ipynb - Linear Regression 📉

#### 🔹 Simple Linear Regression

Predict **sales** using only **TV** advertising spend.

**Key Steps:**

1. Split data: `train_test_split(X, y, test_size=0.2)`
2. Train model: `model.fit(X_train, y_train)`
3. Predict: `y_pred = model.predict(X_test)`
4. Evaluate: R², MAE, MSE, RMSE

**Equation:** `sales = coefficient × TV + intercept`

#### 🔷 Multiple Linear Regression

Predict **sales** using **TV**, **radio**, and **newspaper** advertising.

**Equation:** `sales = c₁×TV + c₂×radio + c₃×newspaper + intercept`

**Evaluation Metrics:**

- 🎯 **R²** - How well model fits (closer to 1 = better)
- 📊 **MAE** - Mean Absolute Error (average prediction error)
- 📉 **MSE** - Mean Squared Error (penalizes large errors)
- 🎲 **RMSE** - Root MSE (same units as target)

---

### advertising.csv 📋

Dataset with 200 rows and 4 columns:

- `TV` - TV advertising budget
- `radio` - Radio advertising budget
- `newspaper` - Newspaper advertising budget
- `sales` - Product sales (target variable)

---

## 🚀 Quick Commands

```python
# Import essentials
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
df = pd.read_csv('advertising.csv')

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
r2 = model.score(X_test, y_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
```

---

## 💡 Tips to Remember

✅ Always split data before training  
✅ Use `random_state=42` for reproducibility  
✅ Multiple regression usually performs better than simple  
✅ Check R² first - it's the easiest metric to interpret  
✅ Lower error metrics (MAE, MSE, RMSE) = better model
