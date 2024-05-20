import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LinearRegression, Ridge # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Load dataset
data = pd.read_csv('kc_house_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Handling missing values (if any)
data = data.dropna()

# Encoding categorical variables (if any)
# For simplicity, let's assume 'zipcode' is the only categorical feature
data = pd.get_dummies(data, columns=['zipcode'], drop_first=True)

# Define features and target variable
X = data.drop(['price', 'id', 'date'], axis=1)  # Drop id and date for simplicity
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Standardize the data (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

# Train models
linear_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)
svr.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_linear = linear_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_svr = svr.predict(X_test_scaled)

# Evaluate models
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

mse_linear, r2_linear = evaluate(y_test, y_pred_linear)
mse_ridge, r2_ridge = evaluate(y_test, y_pred_ridge)
mse_svr, r2_svr = evaluate(y_test, y_pred_svr)

# Print results
print("Linear Regression: MSE = {:.2f}, R² = {:.2f}".format(mse_linear, r2_linear))
print("Ridge Regression: MSE = {:.2f}, R² = {:.2f}".format(mse_ridge, r2_ridge))
print("SVR: MSE = {:.2f}, R² = {:.2f}".format(mse_svr, r2_svr))