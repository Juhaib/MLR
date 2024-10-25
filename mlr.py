import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
data = pd.read_csv('ToyotaCorolla - MLR.csv')

# Step 1: Exploratory Data Analysis (EDA)
print("Data Info:\n", data.info())
print("\nData Summary:\n", data.describe())

# Checking unique values in categorical columns
print("\nUnique values in Fuel_Type:\n", data['Fuel_Type'].unique())
print("Unique values in Automatic:\n", data['Automatic'].unique())

# Step 2: Data Preprocessing

data = pd.get_dummies(data, columns=['Fuel_Type'], drop_first=True)

# Splitting the dataset into features and target variable
X = data.drop(['Price'], axis=1)
y = data['Price']

# Verify the dataset after encoding
print("\nX columns after encoding:\n", X.columns)
print("\nData types after encoding:\n", X.dtypes)

# Define numerical and categorical columns
numeric_features = ['Age_08_04', 'KM', 'HP', 'cc', 'Weight']

# Preprocessor with StandardScaler for numerical features only
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the types in X_train and X_test after splitting
print("\nX_train columns and types:\n", X_train.dtypes)
print("X_test columns and types:\n", X_test.dtypes)

# Step 3: Model Pipeline Setup and Training

# Model 1: Basic Linear Regression with all features
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
print("\nModel 1: Basic Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R²:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))

# Step 4: Apply Lasso and Ridge Regularization
# Lasso Regression
pipeline_lasso = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=0.1))])
pipeline_lasso.fit(X_train, y_train)
y_pred_lasso = pipeline_lasso.predict(X_test)
print("\nLasso Regression")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R²:", r2_score(y_test, y_pred_lasso))
print("MAE:", mean_absolute_error(y_test, y_pred_lasso))

# Ridge Regression
pipeline_ridge = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
pipeline_ridge.fit(X_train, y_train)
y_pred_ridge = pipeline_ridge.predict(X_test)
print("\nRidge Regression")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R²:", r2_score(y_test, y_pred_ridge))
print("MAE:", mean_absolute_error(y_test, y_pred_ridge))