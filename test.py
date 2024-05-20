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

print(melbourne_data.columns)