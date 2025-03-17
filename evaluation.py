#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 

#Importing the Trained Models for Predictions 
from lstm import lstm_train_predict
from randomForest import randomForest_train_predict
from svr import svr_train_predict

#Importing Data
data = pd.read_csv('/Users/carlosmoreno/stock_price_predictor/data/stocks_data.csv')

df = pd.DataFrame(data)