#!/usr/bin/env python
# coding: utf-8

# In[10]:


pip install pandas_datareader


# In[11]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas_datareader.data as pdrd
from datetime import datetime


# In[54]:


#collect data from Yahoo
def collectdata(tickers, start: datetime, end: datetime, data_source, to_csv: bool = True) -> pd.DataFrame:
    prices = []
    for ticker in tickers:
        adj_close = pdrd.DataReader(name=ticker, start=start, end=end, data_source=data_source)[['Adj Close']]
        adj_close.columns = [ticker]
        prices.append(adj_close)
    df_prices = pd.concat(prices, axis=1)
    df_prices = df_prices.reset_index()

    # output csv
    if to_csv:
        path = 'sklearn projects' + datetime.now().strftime('%y-%m-%d_%H%M%S') + '.csv'
        df_prices.to_csv(path, index = True, header=True)
    return df_prices
df_prices = collectdata(tickers=['GOOG', 'MSFT', 'AMZN', 'AAPL', 'NFLX'], 
                        start=datetime(2020,1,2), 
                        end=datetime(2020,12,31), 
                        data_source = 'yahoo')


# In[55]:


df_prices


# In[56]:


df_nf = df_prices[['Date','NFLX']]
df_nf


# In[57]:


plt.figure(figsize = (16,8))
plt.title('Netflex')
plt.xlabel('Days')
plt.ylabel('Close price USD $')
plt.plot(df_nf['Date'], df_nf['NFLX'])
plt.show()


# In[58]:


# Creat a variable to predict 'x' days out into the future

future_days = 25
df_nf['Prediction'] = df_nf['NFLX'].shift(-future_days)
df_nf


# In[61]:


# Get the last 'x' rows data
# df_future = df_nf['NFLX'].tail(future_days)
# df_future
# df_nf = df_nf.dropna()
X = np.array(df_nf.drop(['Date', 'Prediction'], 1))[:-future_days]
X


# In[62]:


y = np.array(df_nf['Prediction'])[:-future_days]
y


# In[63]:


# split the data into 70% training and 25% testing

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[67]:


# Creat the models
# Creat the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)

# Creat linear regression model
Ir = LinearRegression().fit(x_train, y_train)

# Get the last 'x' rows data
x_future = df_nf.drop(['Date', 'Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# In[78]:


# Show the model tree prediction
tree_prediction = tree.predict(x_future)

#Show the model linear agression prediction
Ir_prediction = Ir.predict(x_future)

# Visualize the data
predictions = tree_prediction

valid = df_nf[X.shape[0]:]
valid['Predictions'] = predictions

plt.figure(figsize = (16, 8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD $')
plt.plot(df_nf['NFLX'], color = 'blue')
plt.plot(valid[['NFLX', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])


# In[79]:


# Visualize the data
predictions = Ir_prediction

valid = df_nf[X.shape[0]:]
valid['Predictions'] = predictions

plt.figure(figsize = (16, 8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD $')
plt.plot(df_nf['NFLX'], color = 'blue')
plt.plot(valid[['NFLX', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




