import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import Scale
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression, Ridge # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

def load_data(file_path):
    """Load data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")

def select_features(data, numeric_only=True):
    """Select features from the dataset."""
    if numeric_only:
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        return numeric_columns
    else:
        return data.columns.tolist()

def split_data(X, y, test_size=0.2):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=1)

def standardize_data(train_X, val_X, test_X):
    """Standardize the data for models that require it."""
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)
    return train_X_scaled, val_X_scaled, test_X_scaled

def train_model(model_type, train_X, val_X, train_y, val_y, model_params=None):
    """Train the selected model."""
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Ridge Regression':
        model = Ridge(alpha=model_params.get('alpha', 1.0))
    elif model_type == 'SVR':
        model = SVR(kernel=model_params.get('kernel', 'rbf'),
                    C=model_params.get('C', 100),
                    gamma=model_params.get('gamma', 'scale'),
                    epsilon=model_params.get('epsilon', 0.1))
    else:
        messagebox.showerror("Error", "Invalid model type")
        return None

    model.fit(train_X, train_y)
    return model

def evaluate_model(model, X, y):
    """Evaluate the trained model."""
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    return mae

def display_results(initial_mae, final_mae):
    """Display the evaluation results."""
    result_label.config(text=f"Initial Validation MAE: {initial_mae:,.0f}\nFinal Test MAE: {final_mae:,.0f}")

def train_and_evaluate_model():
    """Train and evaluate the selected model."""
    # Load data
    file_path = file_entry.get()
    data = load_data(file_path)

    # Select features
    numeric_only = numeric_only_var.get()
    features = select_features(data, numeric_only)

    # Select target variable
    target_variable = target_var.get()
    X = data[features]
    y = data[target_variable]

    # Split data
    train_X, val_X, train_y, val_y = split_data(X, y)

    # Standardize data
    train_X_scaled, val_X_scaled, _ = standardize_data(train_X, val_X, val_X)

    # Train initial model
    model_type = model_var.get()
    model_params = {'alpha': alpha_scale.get(),
                    'kernel': kernel_var.get(),
                    'C': C_scale.get(),
                    'gamma': gamma_var.get(),
                    'epsilon': epsilon_scale.get()}
    initial_model = train_model(model_type, train_X, val_X, train_y, val_y, model_params)

    # Evaluate initial model
    if initial_model:
        initial_preds_val = initial_model.predict(val_X)
        initial_mae = mean_absolute_error(val_y, initial_preds_val)

        # Tune model
        final_model = train_model(model_type, train_X_scaled, val_X_scaled, train_y, val_y, model_params)

        # Evaluate final model
        if final_model:
            test_X = scaler.transform(test_X)
            final_preds_test = final_model.predict(test_X)
            final_mae = mean_absolute_error(test_y, final_preds_test)

            # Display results
            display_results(initial_mae, final_mae)

# Create main window
root = tk.Tk()
root.title("House Price Prediction")

# File Entry
file_label = tk.Label(root, text="File Path:")
file_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
file_entry = tk.Entry(root)
file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# Model Selection
model_label = tk.Label(root, text="Model Type:")
model_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
model_var = tk.StringVar()
model_combobox = ttk.Combobox(root, textvariable=model_var, values=['Linear Regression', 'Ridge Regression', 'SVR'])
model_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

# Numeric Features Only Checkbox
numeric_only_var = tk.BooleanVar()
numeric_only_check = tk.Checkbutton(root, text="Numeric Features Only", variable=numeric_only_var)
numeric_only_check.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# Train Button
train_button = tk.Button(root, text="Train and Evaluate", command=train_and_evaluate_model)
train_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Result Label
result_label = tk.Label(root, text="")
result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Start GUI
root.mainloop()