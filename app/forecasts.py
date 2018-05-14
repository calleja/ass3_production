#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 22:51:03 2018

@author: lechuza
"""
import fbprophet
import pandas as pd
import numpy as np
from app import retrieveMarkets as rm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class Forecast(object):
    def __init__(self):
        self.rma=rm.RetrieveMarkets()
        self.two_assets=['ETH','NEO']
    
    def runProgram(self):
        results={'ETH':{},'NEO':{}}
        for i in self.two_assets:
            print('processing {} now'.format(i))
            df=self.prepareDF(i)
            fbprice=self.fbPredict(df)
            lstm_price=self.lstm(df)
            results[i]['fb']='%.6f'%(fbprice)
            results[i]['lstm']='%.6f'%(lstm_price)
        return(results)
        
    def prepareDF(self,ticker):
        mro_df=self.rma.get250Day(ticker)
        print('shape of first df:{}'.format(mro_df.shape))
        mro_df['hi_low_log']=mro_df.apply(lambda x: np.log(x['high']/x['low'])**2,axis=1)

        mro_df['mid']=mro_df.apply(lambda x: np.mean([x['high'],x['low']]),axis=1)

        def parkinson(df,window):
    #return(np.log(max(df['high'])/min(df['low']))**2
            return(np.sqrt((1/(4*np.log(2))*sum(df['hi_low_log']))/window))

        def rolling_apply(df,window):
            i=np.arange(df.shape[0]+1-window)
            results=np.zeros(df.shape[0])
            for g in i:
                results[g+window-1]=parkinson(df.iloc[g:window+g,],window)
            return(results)

        mro_df['vol_3day']=rolling_apply(mro_df,3)
        mro_df['vol_15day']=rolling_apply(mro_df,15)
        roll_2=mro_df[['mid']].rolling(2).mean()
        roll_5=mro_df[['mid']].rolling(5).mean()
        roll_15=mro_df[['mid']].rolling(15).mean()
        roll_2.columns=[i+'_MA2' for i in roll_2.columns]
        roll_5.columns=[i+'_MA5' for i in roll_5.columns]
        roll_15.columns=[i+'_MA15' for i in roll_15.columns]
        aggd=pd.concat([mro_df,roll_2,roll_5,roll_15],axis=1)
        test2=aggd.apply(lambda x: x[8]/x[11],axis=1)
        test5=aggd.apply(lambda x: x[8]/x[12],axis=1)
        test15=aggd.apply(lambda x: x[8]/x[13],axis=1)
        test_df=pd.concat([test2,test5,test15],axis=1)
        var_df=pd.concat([aggd,test_df],axis=1)
        print('we have made it as far as var_df with a shape {}'.format(var_df.shape))
        lista=[x for x in var_df.columns]
        lista[-3:]=['prop2','prop5','prop15']
        var_df.columns=lista
        var_df['mid_ln']=np.log(var_df['mid'])
        var_df['returns']=var_df['mid'].pct_change()
        var_df['ln_diff']=var_df['mid_ln'].diff()
        var_df['std_price']=var_df['mid'].rolling(window=21).std()
        var_df['std_returns']=      var_df['returns'].rolling(window=21).std()
        return(var_df)
        
        
    def fbPredict(self,var_df):
        fb_version=var_df.rename(columns={'time':'ds','mid':'y'})
        print('fb alters the df to a shape {}'.format(fb_version.shape))
#fb_version.dtypes
        ts_prophet=fbprophet.Prophet(changepoint_prior_scale=0.15)
        ts_prophet.fit(fb_version[['y','ds','vol_3day','vol_15day','prop2','prop5','prop15','returns']])

        ts_forecast=ts_prophet.make_future_dataframe(periods=1,freq='D')
        ts_forecast=ts_prophet.predict(ts_forecast)

        return(ts_forecast['yhat'][0])
        
    def lstm(self,var_df):
        var_df1=var_df.dropna(how='any')[['mid','vol_3day','vol_15day','prop2','prop5','prop15']]
#removing 'mid'

#verify this
        all_X=var_df1.iloc[:,1:]
        all_Y=var_df1.iloc[:,0]
        all_Y=all_Y.reshape(all_Y.shape[0],1)


        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(all_X)
        df=pd.DataFrame(scaled,columns=all_X.columns)
        values = df.values

        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled1 = scaler1.fit_transform(all_Y)
        df1=pd.DataFrame(scaled1,columns=['dependent'])
        values1 = df1.values

#we can partition this now... will wait to partition the training set after the performing the lag
        idependent_test=values[-1,:]

#run the lag on only the independent values
        lags=1
        lista=[]
        for i in np.arange(1,lags+1,1):
            temp_df=df.shift(+1,axis=0)
            temp_df.columns=['var'+str(counter+1)+'(t-{})'.format(i) for counter, h in enumerate(temp_df.columns)]
            lista.append(temp_df)

#430 records    
        agg_df=pd.concat(lista,axis=1)
        agg_df['var(t)']=values1
#for QA, compare with var_df1


        agg_df.dropna(how='any',inplace=True)

# split into input and outputs... use all the data available to train
        independent_train=agg_df.values.astype('float32')[:,:-1]
        dependent_train=agg_df.values.astype('float32')[:,-1]
        independent_train.shape

# reshape input to be 3D [samples, timesteps, features]
        train_X = independent_train.reshape((independent_train.shape[0], 1, independent_train.shape[1]))
#while I'm at it, I will transform the out-of-sample data
        test_X = idependent_test.reshape((1,1,idependent_test.shape[0]))


        mdl = Sequential()
        mdl.add(Dense(3, input_shape=(train_X.shape[1],     train_X.shape[2]), activation='relu'))
        mdl.add(LSTM(6, activation='relu'))
        mdl.add(Dense(1, activation='relu'))
        mdl.compile(loss='mean_squared_error', optimizer='adam')
        mdl.fit(train_X, dependent_train, epochs=30, batch_size=1, verbose=0)

#with model in hand, let's fit the prediction
        test_predict = mdl.predict(test_X,verbose=0)
#concatenate with the training dependent:
        y_reunited=np.concatenate((dependent_train,test_predict[0]),axis=0)
        y_reunited=y_reunited.reshape(y_reunited.shape[0],1)

#inverse transform the output and get in same units as the input

        inv_agg_data=scaler1.inverse_transform(y_reunited)
        return(inv_agg_data[-1,:][0])
