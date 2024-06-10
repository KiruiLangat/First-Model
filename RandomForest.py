from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from learntools.core import binder
from learntools.machine_learning.ex6 import *
import pandas as pd

# Path of the file to read
mobile_file_path = 'C:/Users/sms20/Machine Learning/First-Model/train.csv'

home_data = pd.read_csv(mobile_file_path)

# Create target object and call it y
y = home_data.price_range

# Create X
features = ['battery_power', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'int_memory', 'wifi']

X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define model
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print(f"Validation MAE for Random Forest Model: {rf_val_mae}")
