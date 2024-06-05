import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import binder
from learntools.machine_learning.ex5 import *

# Path of the file to read
iowa_file_path = 'C:/Users/sms20/Machine Learning/First-Model/melb_data.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y

print(home_data.columns)
print(home_data.head())
home_data.dropna(axis=0)

# y = home_data.Price

# # Create X
# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# X = home_data[features]

# # Split into validation and training data
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# # Specify Model
# iowa_model = DecisionTreeRegressor(random_state=1)

# # Fit Model
# iowa_model.fit(train_X, train_y)

# # Make validation predictions and calculate mean absolute error
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(val_predictions, val_y)
# print("Validation MAE: {:,.0f}".format(val_mae))

# # Set up code checking
# binder.bind(globals())
# print("\nSetup complete")
