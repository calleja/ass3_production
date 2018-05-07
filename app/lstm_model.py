#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:45:29 2018

@author: lechuza
"""
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd


lags=1 #applying three lags
var_df1=var_df.dropna(how='any')[['mid','vol_3day','vol_15day','prop2','prop5','prop15']]
#removing 'mid'
temp_df=var_df1.iloc[:,1:]

#verify this
test_X=var_df1.iloc[-1,1:]
test_X
lista=[]
for i in np.arange(1,lags+1,1):
    temp_df=temp_df.shift(+1,axis=0)
    temp_df.columns=['var'+str(counter+1)+'(t-{})'.format(i) for counter, h in enumerate(temp_df.columns)]
    lista.append(temp_df)

#will need to concatenate the last row, ie test_X to agg_df in order to transform the data
agg_df=pd.concat(lista,axis=1)

pd.concat(agg_df,test_X

var_df1.tail(2)
agg_df.tail(2)

#may not want to do this step
agg_df['var(t)']=var_df1['mid']

#will ditch the first row
agg_df.dropna(how='any',inplace=True)
agg_arr=agg_df.values.astype('float32')

agg_arr[-1]
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(agg_arr[:,:-1])

agg_arr[-1,:]

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(y_values)




df=pd.DataFrame(scaled,columns=agg_df.columns)
df.tail()
values = df.values
# split into input and outputs... use all the data available to train
train_X, train_y = values[:, :-1], values[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

mdl = Sequential()
mdl.add(Dense(3, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu'))
mdl.add(LSTM(6, activation='relu'))
mdl.add(Dense(1, activation='relu'))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(train_X, train_y, epochs=30, batch_size=1, verbose=0)

#defined above
#treat test_X:
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
test_predict = mdl.predict(test_X,verbose=0)

#inverse transform the output and get in same units as the input
#first will aggregate the train and test sets - appending the prediction to the dependent values vector:
test_X1 = test_X.reshape((test_X.shape[0], test_X.shape[2]))
y_train=np.reshape(train_y,(train_y.shape[0],1))
train_data = np.concatenate((train_X,y_train), axis=1)
test_data = np.concatenate((test_X1,test_predict), axis=1)
agg_data = np.concatenate((train_data,test_data), axis=0)
inv_agg_data=scaler.inverse_transform(agg_data)
inv_agg_data[-1,-1]