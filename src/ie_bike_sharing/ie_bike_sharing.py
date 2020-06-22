#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries required for the project 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import datetime
import datetime as dt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from pprint import pprint
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn import svm
import pickle
from pathlib import Path
import joblib


# In[2]:


def clean_and_dummify(input_data):
    
    if type(input_data) == str:
        input_data = pd.read_csv(input_data)
    
    #DROP COLUMNS WE KNOW WE WON'T USE
    input_data = input_data.drop(columns = ['dteday', 'casual','atemp', 'registered'])
    
    #PREPROCESS DATA:
    
    #creating duplicate columns for feature engineering
    input_data['hr2'] = input_data['hr']
    input_data['season2'] = input_data['season']
    input_data['temp2'] = input_data['temp']
    input_data['hum2'] = input_data['hum']
    input_data['weekday2'] = input_data['weekday']

    
    # Convert the data type to eithwe category or to float
    int_hour = ["season","yr","mnth","hr","holiday","weekday","workingday","weathersit"]
    for col in int_hour:
        input_data[col] = input_data[col].astype("category")
        
    #handle skewness
    input_data["windspeed"] = np.log1p(input_data.windspeed)
    
    #for train data only
    if 'cnt' in input_data.columns:
        input_data["cnt"] = np.sqrt(input_data.cnt)
    
    #FEATURE ENGINEERING
    
    #Rented during office hours
    input_data['IsOfficeHour'] = np.where((input_data['hr2'] >= 9) & (input_data['hr2'] < 17) & (input_data['weekday2'] == 1), 1 ,0)
    input_data['IsOfficeHour'] = input_data['IsOfficeHour'].astype('category')
    
    #Rented during daytime
    input_data['IsDaytime'] = np.where((input_data['hr2'] >= 6) & (input_data['hr2'] < 22), 1 ,0)
    input_data['IsDaytime'] = input_data['IsDaytime'].astype('category')
    
    #Rented during morning rush hour
    input_data['IsRushHourMorning'] = np.where((input_data['hr2'] >= 6) & (input_data['hr2'] < 10)  & (input_data['weekday2'] == 1), 1 ,0)
    input_data['IsRushHourMorning']=input_data['IsRushHourMorning'].astype('category')
    
    #Rented during evening rush hour
    input_data['IsRushHourEvening'] = np.where((input_data['hr2'] >= 15) & (input_data['hr2'] < 19) & (input_data['weekday2'] == 1), 1 ,0)
    input_data['IsRushHourEvening'] = input_data['IsRushHourEvening'].astype('category')
    
    #Rented during most busy season
    input_data['IsHighSeason'] = np.where((input_data['season2'] == 3), 1 ,0)
    input_data['IsHighSeason'] = input_data['IsHighSeason'].astype('category')
    
    #binning temp and hum in 5 equally sized bins
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    input_data['temp_binned'] = pd.cut(input_data['temp2'], bins).astype('category')
    input_data['hum_binned'] = pd.cut(input_data['hum2'], bins).astype('category')
    
    #dropping duplicated rows used for feature engineering
    cleaned_dataset = input_data.drop(columns = ['hr2','season2', 'temp2', 'hum2', 'weekday2'])
    
    #dummify
    cleaned_dummified_data = pd.get_dummies(cleaned_dataset)
    
    return cleaned_dummified_data


# In[3]:


def train_and_persist(training_data): 
    
    #READ THE DATA
    if type(training_data) == str:
        training_data_df = pd.read_csv(training_data)
    
    #CLEAN THE  TRAIN DATA
    cleaned_training_data = clean_and_dummify(training_data_df.iloc[0:15211])
    
    # seperate the independent and target variable on testing data
    train_X = cleaned_training_data.drop(columns=['cnt'],axis=1)
    train_y = cleaned_training_data['cnt']
    
    model = RandomForestRegressor(max_depth = 40,
                           min_samples_leaf = 1,
                           min_samples_split = 2,
                           n_estimators= 200,
                           random_state = 42)
    model.fit(train_X, train_y)
    
    #PERSISTENCE
    
    from joblib import dump, load
    saved_model = dump(model, (str(Path.home())+'\model.pkl'))
    retrieved_model = joblib.load(str(Path.home())+'\model.pkl')
    
    return retrieved_model


# In[9]:


#test the functions by inputting the following parameters to the model

parameters = ({
    "instant": 2,
    "dteday": '1/1/2011',
    "season": 1,
    "yr": 0,
    "mnth": 1,
    "hr": 0,
    "holiday": 0,
    "weekday": 6,
    "workingday": 0,
    "weathersit": 1,
    "temp": 0.22,
    "atemp": 0.2727,
    "hum": 0.8,
    "windspeed": 0,
    "casual": 8,
    "registered": 32
 })


# In[5]:


#ensure input parameters match model's parameters

master_columns_list = clean_and_dummify('hour.csv').columns.tolist()
master2 = master_columns_list.copy()
master2.remove('cnt')


# In[6]:


def predict(parameters):
    #convert dictionary to dataframe and apply preprocessing
    test_df = pd.DataFrame(parameters, index=[0])
    cleaned_test_df = clean_and_dummify(test_df)
    
    for col in master_columns_list:
        if (col not in cleaned_test_df.columns) & (col != 'cnt'):
            cleaned_test_df[col] = 0
            
    #REORDER COLUMNS:
    cleaned_test_df = cleaned_test_df[master2]
    
    if 'model.pkl' in os.listdir(Path.home()):
        model = joblib.load(str(Path.home())+'\model.pkl')
        output = model.predict(cleaned_test_df)
    
    else:
        model = train_and_persist('hour.csv')
        output = model.predict(cleaned_test_df)
        
    return output


# In[10]:


predict(parameters)

