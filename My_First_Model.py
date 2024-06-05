import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import binder
from learntools.machine_learning.ex5 import *

# Path of the file to read
mobile_file_path = 'C:/Users/sms20/Machine Learning/First-Model/train.csv'

home_data = pd.read_csv(mobile_file_path)

print(home_data.columns)
print(home_data.head())
print(home_data.price_range)

# Create target object and call it y
y = home_data.price_range

# # Create X
features = ['battery_power', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'int_memory', 'wifi']

X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# # Specify Model
mobile_model = DecisionTreeRegressor(random_state=1)
# # Fit Model
mobile_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = mobile_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# Set up code checking
binder.bind(globals())
print("\nSetup complete")

# function 
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Loop to compare tree sizes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)

# Fit the model with best_tree_size. Fill in argument to make optimal size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model
final_model.fit(X, y)