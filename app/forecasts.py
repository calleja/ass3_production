#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 22:51:03 2018

@author: lechuza
"""
import sys
import fbprophet
import pandas as pd
import numpy as np
from app import retrieveMarkets as rm

sys.path.append('/home/lechuza/Documents/CUNY/data_607/flask_work/crypto_flask')

class Forecast(object):
    def __init__(self,rma):
        self.rma=rma
        self.two_assets=['XMR','ETH']
    
    def runProgram(self):
        results={}
        for i in self.two_assets:
            df=self.prepareDF(i)
            fbprice=self.fbPredict(df)
            results[i]=[fbprice]
        
    def prepareDF(self,ticker):
        mro_df=self.rma.get250Day(ticker)
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

#fb_version.dtypes
        ts_prophet=fbprophet.Prophet(changepoint_prior_scale=0.15)
        ts_prophet.fit(fb_version[['y','ds','vol_3day','vol_15day','prop2','prop5','prop15','returns']])

        ts_forecast=ts_prophet.make_future_dataframe(periods=14,freq='D')
        ts_forecast=ts_prophet.predict(ts_forecast)

        return(ts_forecast['yhat'][var_df.shape[0]])
