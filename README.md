# House Price Prediction GUI

This project predicts the selling price of houses based on features such as square footage, number of bedrooms and bathrooms, location, amenities, etc. It allows the user to choose between different regression models (Linear Regression, Ridge Regression, and SVR) and select the features to be used in the prediction through a graphical user interface (GUI).

## Features

- Dropdown menu to select the regression model.
- Checkbox selection for features to be included in the model.
- Button to train the model and display results.
- Displays initial validation mean absolute error (MAE) and final test MAE after tuning.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Required Libraries

Install the required Python libraries using the following command:

```sh
pip install pandas scikit-learn tk
```

## Dataset ##
The dataset used in this project is melb_data.csv. Ensure that this file is in the same directory as the code.

## Running Instructions ##
1. Clone this repository or download the code.
2. Ensure that the melb_data.csv and kc_house.csv files are present in the same directory as the code.
3. Run the Python script to start the GUI:
   ```sh
    python code_kc.py
   ```
   or
   ```sh
    python code_melb.py
   ```
   or
   ```sh
    python code_melb_gui.py
   ```
  
## GUI Instructions ##
1. Model Selection: Use the dropdown menu to select the type of regression model you want to use (Linear Regression, Ridge Regression, SVR).
2. Feature Selection: Check the boxes next to the features you want to include in your model. Only numeric features are available for selection.
3. Train Model: Click the "Train Model" button to train the selected model with the chosen features.
4. Results: The initial validation MAE and final test MAE after tuning (if applicable) will be displayed in the text area below the "Train Model" button.

## Acknowledgments ##
+ The dataset used in this project is from the Melbourne Housing Market dataset available on Kaggle.
+ The scikit-learn library was used for implementing the machine learning models.
+ Tkinter was used for building the graphical user interface.
