#!/usr/bin/env python
# coding: utf-8

# # Predicting Sales

# In[1]:


from datetime import datetime, timedelta,date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from __future__ import division


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


# In[5]:


import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[6]:


pyoff.init_notebook_mode()


# In[7]:


df_sales = pd.read_csv('train.csv')


# In[8]:


df_sales.shape


# In[9]:


df_sales.head(10)


# In[10]:


df_sales['date'] = pd.to_datetime(df_sales['date'])


# In[11]:


#represent month in date field as its first day
df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
df_sales['date'] = pd.to_datetime(df_sales['date'])


# In[12]:


#groupby date and sum the sales
df_sales = df_sales.groupby('date').sales.sum().reset_index()


# In[13]:


df_sales.head()


# In[14]:


#plot monthly sales
plot_data = [
    go.Scatter(
        x=df_sales['date'],
        y=df_sales['sales'],
    )
]

plot_layout = go.Layout(
        title='Montly Sales'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[15]:


#create a new dataframe to model the difference
df_diff = df_sales.copy()


# In[16]:


#add previous sales to the next row
df_diff['prev_sales'] = df_diff['sales'].shift(1)


# In[17]:


df_diff.head()


# In[18]:


#drop the null values and calculate the difference
df_diff = df_diff.dropna()


# In[19]:


df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])


# In[20]:


df_diff.head(10)


# # Let’s plot it and check if it is stationary now

# In[21]:


#plot sales diff
plot_data = [
    go.Scatter(
        x=df_diff['date'],
        y=df_diff['diff'],
    )
]

plot_layout = go.Layout(
        title='Montly Sales Diff'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[22]:


#create new dataframe from transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'],axis=1)


# In[23]:


#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)


# In[24]:


df_supervised.head(10)


# In[25]:


df_supervised.tail(6)


# In[26]:


#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)


# In[27]:


# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1', data=df_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# In[28]:


# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5', data=df_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# In[29]:


# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# In[30]:


#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['sales','date'],axis=1)


# In[31]:


#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values


# In[32]:


df_model.info()


# In[33]:


#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)

# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)


# # Building the LSTM model

# In[34]:


X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])


# In[35]:


X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# In[37]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False)


# In[38]:


y_pred = model.predict(X_test,batch_size=1)


# In[39]:


y_pred


# In[40]:


y_test


# In[41]:


#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])


# In[43]:


#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print (np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))


# In[44]:


pred_test_set[0]


# In[45]:


#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])


# In[46]:


#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)


# # create dataframe that shows the predicted sales

# In[47]:


#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-7:].date)
act_sales = list(df_sales[-7:].sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)


# In[48]:


df_result


# In[49]:


df_sales.head()


# In[50]:


#merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales,df_result,on='date',how='left')


# In[51]:


df_sales_pred


# In[52]:


#plot actual and predicted
plot_data = [
    go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['sales'],
        name='actual'
    ),
        go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['pred_value'],
        name='predicted'
    )
    
]

plot_layout = go.Layout(
        title='Sales Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:




