import numpy as np
import pandas as pd

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

# load the csv file 
data = pd.read_csv('outputs/predictions.csv')

# convert columns to arrays 
y_true = np.array(data['Actual_Price'])
y_pred = np.array(data[' Predicted_Price'])

#
print("MAE:", mae(y_true, y_pred))
print("MAPE:", mape(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))
print("SMAPE:", smape(y_true, y_pred))