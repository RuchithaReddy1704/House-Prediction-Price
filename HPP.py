# Step 1: Import Libraries 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

# Step 2: Load Dataset 

data = fetch_california_housing() 
df = pd.DataFrame(data.data, columns=data.feature_names) 
df['Price'] = data.target 

# Step 3: Explore & Clean Data 

print(df.head()) 
print(df.isnull().sum()) 
sns.pairplot(df[['MedInc', 'AveRooms', 'AveOccup', 'Price']]) 
plt.show() 

# Step 4: Feature Selection 

correlation_matrix = df.corr() 
plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') 
plt.title("Feature Correlation Heatmap") 
plt.show() 

# Drop features with high correlation or low importance (if any) 
# In this example, we keep all for simplicity 

# Step 5: Split Data 

X = df.drop('Price', axis=1) 
y = df['Price'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Step 6: Train Models 

# Linear Regression 
lr = LinearRegression() 
lr.fit(X_train, y_train) 

# Random Forest 
rf = RandomForestRegressor(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train) 

# XGBoost 
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1) 
xgb.fit(X_train, y_train) 

# Step 7: Evaluate Models 

def evaluate_model(model, name): 
    preds = model.predict(X_test) 
    print(f"--- {name} ---") 
    print("R2 Score:", r2_score(y_test, preds)) 
    print("MAE:", mean_absolute_error(y_test, preds)) 
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds))) 
    print() 

evaluate_model(lr, "Linear Regression") 
evaluate_model(rf, "Random Forest") 
evaluate_model(xgb, "XGBoost") 

# Bonus: Visualize Actual vs Predicted 

plt.figure(figsize=(8, 5)) 
plt.scatter(y_test, rf.predict(X_test), alpha=0.6, color='green') 
plt.xlabel("Actual Prices") 
plt.ylabel("Predicted Prices") 
plt.title("Random Forest: Actual vs Predicted") 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
plt.show()
