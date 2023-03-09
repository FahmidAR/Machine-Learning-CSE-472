# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# load the data into a pandas dataframe
df = pd.read_csv('GP.csv')

X=[]
Y=[]
y = df['Close'].values


# Take the first 5 rows of y and make the sixth row the target
for i in range(y.shape[0]-6):
    X.append(y[i:i+5])
    Y.append(y[i+5])

X = np.array(X)
Y = np.array(Y)

split_ratio = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1-split_ratio, random_state=42)

# create a list of regression models
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    KNeighborsRegressor(),
    MLPRegressor()
]

best_model = None
best_rmse = float('inf')

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model {i+1} RMSE: {rmse:.2f}")
    print(f"Model {i+1} MSE: {mse:.2f}")
    
    # find the best model
    if rmse < best_rmse:
        best_model = model
        best_rmse = rmse

print(f"Best model: {type(best_model).__name__} (RMSE: {best_rmse:.2f})")



print("Last 5 business days' CP need to be entered to predict the next business day's CP")
day_5 = float(input("Enter CP of Day 5 : "))
day_4 = float(input("Enter CP of Day 4 : "))
day_3 = float(input("Enter CP of Day 3 : "))
day_2 = float(input("Enter CP of Day 2 : "))
day_1 = float(input("Enter CP of Day 1 : "))


# Loop through each model in the list of models
for model in models:

    # Create a numpy array containing the last five business days' CP entered by the user
    input_data = np.array([day_5, day_4, day_3, day_2, day_1]).reshape(1, -1)

    # Train the model using the training data X and labels Y
    model.fit(X, Y)

    # Use the trained model to predict the CP for the next business day based on the input data
    next_day_cp = model.predict(input_data)[0]

    # Print the predicted CP along with the name of the model used for the prediction
    model_name = type(model).__name__
    print(f"Next day's CP: {next_day_cp:.2f} (predicted by {model_name})")
