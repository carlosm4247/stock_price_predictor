# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 

# Importing the Trained Models for Predictions 
from lstm import lstm_train_predict
from randomForest import randomForest_train_predict
from svr import svr_train_predict

# Importing Data
data = pd.read_csv('/Users/carlosmoreno/stock_price_predictor/data/stocks_data.csv')

df = pd.DataFrame(data)

# Evaluation Function
def evaluate_model(model, stock, Y_test, Y_predictions):
    print(f"{model} evaluation for {stock}:")

    mse = mean_squared_error(Y_test, Y_predictions)
    r2 = r2_score(Y_test, Y_predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return r2, mse