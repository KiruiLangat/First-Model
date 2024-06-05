# My First Model
This Python script uses the Decision Tree Regressor model from the Scikit-learn library to predict house prices. 
The model is trained on a dataset of Mobile Price Classification.

### Libraries Used
- pandas: For data manipulation and analysis.
- sklearn.metrics: For calculating the mean absolute error of the model.
- sklearn.model_selection: For splitting the dataset into training and validation sets.
- sklearn.tree: For using the Decision Tree Regressor model.
- learntools.core: For setting up code checking.
- learntools.machine_learning.ex5: For additional machine learning tools.

### Data
The data is read from a CSV file located at my local host downloaded at Kaggle.

### Model
The model used is a Decision Tree Regressor. 
The model is specified with a random state of 1 for reproducibility. 
The model is then fitted with the training data.

### Validation and Error Calculation
The model makes predictions on the validation data and the mean absolute error between the predictions and the actual values is calculated. 
The mean absolute error is then printed to the console.

### Code Checking
The script sets up code checking with the binder.bind(globals()) command. This ensures that the script is running correctly.


More to be done and updated.
Forever Learning ♾

