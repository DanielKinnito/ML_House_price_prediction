import tkinter as tk
from tkinter import ttk
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression, Ridge # type: ignore
from sklearn.svm import SVR# type: ignore
from sklearn.metrics import mean_absolute_error# type: ignore
from sklearn.preprocessing import StandardScaler# type: ignore

class HousePricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")

        self.model_options = ['Linear Regression', 'Ridge Regression', 'SVR']
        self.selected_model = tk.StringVar(value=self.model_options[0])

        self.load_data()
        self.create_widgets()

    def load_data(self):
        # Load dataset
        self.melbourne_data = pd.read_csv('melb_data.csv')
        # Drop rows with missing values
        self.melbourne_data = self.melbourne_data.dropna(axis=0)

    def create_widgets(self):
        # Model Selection
        model_label = ttk.Label(self.root, text="Select Model:")
        model_label.grid(row=0, column=0, sticky=tk.W)

        model_dropdown = ttk.Combobox(self.root, textvariable=self.selected_model, values=self.model_options, state="readonly")
        model_dropdown.grid(row=0, column=1, padx=10, pady=10)

        # Feature Selection
        feature_label = ttk.Label(self.root, text="Select Features:")
        feature_label.grid(row=1, column=0, sticky=tk.W)

        self.features = self.melbourne_data.select_dtypes(include='number').columns.tolist()
        self.selected_features = []

        for i, feature in enumerate(self.features):
            feature_var = tk.IntVar(value=1)
            checkbox = ttk.Checkbutton(self.root, text=feature, variable=feature_var)
            checkbox.grid(row=i+2, column=0, columnspan=2, sticky=tk.W)
            self.selected_features.append((feature, feature_var))

        # Train Model Button
        train_button = ttk.Button(self.root, text="Train Model", command=self.train_model)
        train_button.grid(row=len(self.features)+3, column=0, columnspan=2, pady=10)

        # Results Display
        self.results_text = tk.Text(self.root, height=10, width=50)
        self.results_text.grid(row=len(self.features)+4, column=0, columnspan=2, padx=10, pady=10)

    def train_model(self):
        # Extract selected features
        selected_features = [feature for feature, var in self.selected_features if var.get() == 1]

        # Extract target variable
        y = self.melbourne_data['Price']

        # Extract features
        X = self.melbourne_data[selected_features]

        # Split the data into training and testing sets
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

        # Standardize the data for models that require it
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)

        # Train the initial model
        initial_model = self.get_selected_model(train_X, train_y)
        initial_model.fit(train_X, train_y)
        initial_preds = initial_model.predict(test_X)
        initial_mae = mean_absolute_error(test_y, initial_preds)

        # Tune the model
        tuned_model, tuned_mae = self.tune_model(initial_model, train_X_scaled, train_y, test_X_scaled, test_y)

        # Display results
        self.display_results(initial_mae, tuned_mae)

    def get_selected_model(self, X, y):
        if self.selected_model.get() == 'Linear Regression':
            return LinearRegression()
        elif self.selected_model.get() == 'Ridge Regression':
            return Ridge(alpha=1.0)
        elif self.selected_model.get() == 'SVR':
            return SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

    def tune_model(self, model, train_X, train_y, test_X, test_y):
        if self.selected_model.get() == 'Linear Regression':
            return model, None
        elif self.selected_model.get() == 'Ridge Regression':
            alpha_values = [0.1, 1.0, 10.0, 100.0]
            ridge_mae_scores = {alpha: self.get_ridge_mae(alpha, train_X, train_y, test_X, test_y) for alpha in alpha_values}
            best_alpha = min(ridge_mae_scores, key=ridge_mae_scores.get)
            return Ridge(alpha=best_alpha), ridge_mae_scores[best_alpha]
        elif self.selected_model.get() == 'SVR':
            C_values = [1, 10, 100]
            gamma_values = ['scale', 'auto']
            epsilon_values = [0.1, 0.01, 0.001]
            svr_mae_scores = {(C, gamma, epsilon): self.get_svr_mae(C, gamma, epsilon, train_X, train_y, test_X, test_y)
                            for C in C_values for gamma in gamma_values for epsilon in epsilon_values}
            best_params = min(svr_mae_scores, key=svr_mae_scores.get)
            return SVR(kernel='rbf', C=best_params[0], gamma=best_params[1], epsilon=best_params[2]), svr_mae_scores[best_params]

    def get_ridge_mae(self, alpha, train_X, train_y, test_X, test_y):
        model = Ridge(alpha=alpha)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        mae = mean_absolute_error(test_y, preds)
        return mae

    def get_svr_mae(self, C, gamma, epsilon, train_X, train_y, test_X, test_y):
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        mae = mean_absolute_error(test_y, preds)
        return mae

    def display_results(self, initial_mae, tuned_mae):
        self.results_text.delete(1.0, tk.END)  # Clear previous results

        # Initial Model Results
        self.results_text.insert(tk.END, "Initial Model Results:\n")
        self.results_text.insert(tk.END, f"Initial Test MAE: {initial_mae:.0f}\n")

        # Tuned Model Results (if applicable)
        if tuned_mae is not None:
            self.results_text.insert(tk.END, "\nTuned Model Results:\n")
            self.results_text.insert(tk.END, f"Tuned Test MAE: {tuned_mae:.0f}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricePredictorGUI(root)
    root.mainloop()