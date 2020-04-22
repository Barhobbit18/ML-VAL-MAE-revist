#Validation Mean Absolute Error (MAE)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

file = 'train.csv'

data = pd.read_csv(file)

# Create target Y
y = data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = data[features]

# Split into validation and training data
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Specify Model
model = DecisionTreeRegressor(random_state=1)
# Fit Model
model.fit(train_x, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = model.predict(val_x)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation Mean Absolute Error (MAE) not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

#Best value Max nodes########################################################
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_x, val_x, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
##################################################################
# Using best value for max_leaf_nodes
model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
model.fit(train_x, train_y)
val_predictions = model.predict(val_x)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation Mean Absolute Error (MAE) best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define &Set random_state to 1
Ran_Forest_model = RandomForestRegressor(random_state=1)
Ran_Forest_model.fit(train_x, train_y)
Ran_Forest_val_predictions = Ran_Forest_model.predict(val_x)
Ran_Forest_val_mae = mean_absolute_error(Ran_Forest_val_predictions, val_y)

print("Validation Mean Absolute Error (MAE)for Random Forest Model: {:,.0f}".format(Ran_Forest_val_mae))
