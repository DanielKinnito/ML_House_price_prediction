import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression, Ridge # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Load dataset
melbourne_data = pd.read_csv('melb_data.csv')

# Dropping data where there are missing info
melbourne_data = melbourne_data.dropna(axis=0)

# Prediction target
y = melbourne_data.Price

# Choosing features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'YearBuilt', 'Distance', 'Propertycount', 'Bedroom2', 'Car', 'BuildingArea']
X = melbourne_data[melbourne_features]

# Split the data into training, validation, and final testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=1)

# Standardize the data for models that require it
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)
test_X_scaled = scaler.transform(test_X)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(train_X, train_y)
linear_preds_val = linear_model.predict(val_X)
linear_val_mae = mean_absolute_error(val_y, linear_preds_val)
print("Linear Regression Initial Validation MAE: {:,.0f}".format(linear_val_mae))

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(train_X, train_y)
ridge_preds_val = ridge_model.predict(val_X)
ridge_val_mae = mean_absolute_error(val_y, ridge_preds_val)
print("Ridge Regression Initial Validation MAE: {:,.0f}".format(ridge_val_mae))

# Support Vector Regression
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(train_X_scaled, train_y)
svr_preds_val = svr_model.predict(val_X_scaled)
svr_val_mae = mean_absolute_error(val_y, svr_preds_val)
print("SVR Initial Validation MAE: {:,.0f}".format(svr_val_mae))

# Function to tune Linear Regression model
def get_linear_regression_mae(train_X, val_X, train_y, val_y):
    model = LinearRegression()
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae, model

# Function to tune Ridge Regression model
def get_ridge_mae(alpha, train_X, val_X, train_y, val_y):
    model = Ridge(alpha=alpha)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae, model

# Function to tune SVR model
def get_svr_mae(C, gamma, epsilon, train_X, val_X, train_y, val_y):
    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae, model

# Tune Linear Regression
linear_mae, final_linear_model = get_linear_regression_mae(train_X, val_X, train_y, val_y)
linear_preds_test = final_linear_model.predict(test_X)
linear_test_mae = mean_absolute_error(test_y, linear_preds_test)
print("\nLinear Regression Final Test MAE: {:,.0f}".format(linear_test_mae))

# Tune Ridge Regression
alpha_values = [0.1, 1.0, 10.0, 100.0]
ridge_mae_scores = {alpha: get_ridge_mae(alpha, train_X, val_X, train_y, val_y)[0] for alpha in alpha_values}
best_ridge_alpha = min(ridge_mae_scores, key=ridge_mae_scores.get)

final_ridge_mae, final_ridge_model = get_ridge_mae(best_ridge_alpha, train_X, val_X, train_y, val_y)
ridge_preds_test = final_ridge_model.predict(test_X)
ridge_test_mae = mean_absolute_error(test_y, ridge_preds_test)
print("Ridge Regression Final Test MAE: {:,.0f}".format(ridge_test_mae))

# Tune SVR
C_values = [1, 10, 100]
gamma_values = ['scale', 'auto']
epsilon_values = [0.1, 0.01, 0.001]
svr_mae_scores = {(C, gamma, epsilon): get_svr_mae(C, gamma, epsilon, train_X_scaled, val_X_scaled, train_y, val_y)[0]
                  for C in C_values for gamma in gamma_values for epsilon in epsilon_values}
best_svr_params = min(svr_mae_scores, key=svr_mae_scores.get)

final_svr_mae, final_svr_model = get_svr_mae(*best_svr_params, train_X_scaled, val_X_scaled, train_y, val_y)
svr_preds_test = final_svr_model.predict(test_X_scaled)
svr_test_mae = mean_absolute_error(test_y, svr_preds_test)
print("SVR Final Test MAE: {:,.0f}".format(svr_test_mae))