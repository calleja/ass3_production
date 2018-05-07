#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Will need to retrieve universe of available currencies from bittrex and place into a pd df
"""
from matplotlib import pyplot as plt
import pandas as pd
import requests
import datetime
import numpy as np


class RetrieveMarkets():
    #some global variables for Bittrex
    
    def __init__(self):
        self.api_key='c8f16ead26d4438e991c318c8fd76629'
        self.api_secret='bb885473805547f1834cfc5057a78901'
        self.base_currency='BTC'        

    def getCurrencies(self):
        
        url='https://bittrex.com/api/v1.1/public/getcurrencies'
        #all arguments must be converted to string
        payload={'apikey':self.api_key,
'apisecret':self.api_secret,'nonce':datetime.datetime.now()}
        r=requests.get(url,params=payload)
        df=pd.DataFrame(r.json()['result'])
        self.df_active=df[df.IsActive == True]
    #display this dataframe to the user
        return(self.df_active.loc[:,['Currency','CurrencyLong']])


    def getCurrentPrice(self,ticker_list):
        #consumed by engageUser in the selectExecPrice() which itself is called by eu.prepareTrade()
        url_current='https://bittrex.com/api/v1.1/public/getticker'
#must be a BTC based cross market
        #a list of nested dictionaries
        all_prices_dict={}
        for single_tick in ticker_list:
            self.market=self.base_currency+'-'+single_tick
            print('Sending over the following currency pair to the bittrex api: {}'.format(self.market))
            #print('sending the trading pair {} to bittrex'.format(self.market))
            payload2={'apikey':self.api_key,
'apisecret':self.api_secret,'nonce':datetime.datetime.now(),'market':self.market}
            r=requests.get(url_current,params=payload2) 
            print('result from the api: {}'.format(r.json()['result']))
            price_dict=r.json()['result']
            #add the ticker associated w/prices to dict... AS ITS KEY... dict_list will be a list of nested dictionaries
            #TODO have the option of making this a dictionary of dictionaries instead of a list of dictionaries, which is more difficult to index
            #print('response from bittrex {}'.format(price_dict))
            all_prices_dict[single_tick]=price_dict
        #returns the ask, bid, last trades in a dict/json document... this will be refactored to return a list of dictionaries 
        return(all_prices_dict)
    
    def getCurrentPriceCC(self,ticker_list):
        #bittrex API stopped working, interact with cryptocompare
        url_now='https://min-api.cryptocompare.com/data/price'
        #fsym=ETH&tsyms=BTC,USD,EUR'
        self.stats_dic={}
        for single_tick in ticker_list:
            #look up the ticker from the index number
            #instantiate the subdicitonary
            payload2={'tsyms':'BTC','fsym':single_tick, 'apikey':self.api_key,'apisecret':self.api_secret,'nonce':datetime.datetime.now()}
            r=requests.get(url_now,params=payload2) 
            print('printing the retrieved market price from retrieveMarkets object {}'.format(r.json()))
            for key, value in r.json().items():
                self.stats_dic[key]=value
            return(self.stats_dic)
            
    def get24Hr(self, ticker_list):
        url_24_hr='https://min-api.cryptocompare.com/data/histohour?'
        self.stats_dic={}
    
        for single_tick in ticker_list:
            #look up the ticker from the index number
            #instantiate the dicitonary
            self.stats_dic[single_tick]={}
            payload2={'apikey':self.api_key,
'apisecret':self.api_secret,'nonce':datetime.datetime.now(),'fsym':single_tick,'tsym':'BTC','limit':24}
            r=requests.get(url_24_hr,params=payload2) 
            df=pd.DataFrame.from_dict(r.json()['Data'])
            df_mat=df.as_matrix()
            self.stats_dic[single_tick]['max']=np.amax(df_mat[:,1])
            self.stats_dic[single_tick]['min']=np.amin(df_mat[:,1])
            self.stats_dic[single_tick]['avg']=np.average(df_mat[:,1])
        return(self.stats_dic)
    
    def get100Day(self,ticker):
        ''' Acquire historical prices from CRYPTOCOMPARE '''
        url='https://min-api.cryptocompare.com/data/histoday'
        #will return a str
        parameters= {'fsym':ticker, 'tsym': self.base_currency, 'e': 'Bittrex', 'aggregate':1,'limit':120}
        
        r=requests.get(url,parameters)
        #handle this error in the calling class
        j_obj=r.json()
        if j_obj['Response']=='Error':
            print("fetch didn't work")
            raise RuntimeError
        raw_time=j_obj['Data']
        df=pd.DataFrame.from_dict(raw_time)
        df['time']=df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        #print the price chart... returns the path to the image for html rendering
        return(self.draw100day(df,ticker))
    
    
    def draw100day(self,df,ticker):
        #calculate the date 100 days ago
        day_100=datetime.datetime.now() - datetime.timedelta(days=100)
        df['ma_20']=df['close'].rolling(20).mean()
        
        plt.subplot(1,1,1)
        plt.xticks(rotation=45)
        plt.xlim(day_100,datetime.datetime.now())
        #plt.plot(df['time'],df['close'],color='red',marker='o')
        plt.plot(df['time'],df['close'],color='green',linestyle='-')
        plt.plot(df['time'],df['ma_20'],color='cyan',linestyle='-')
#plt.title(str(self.market)+' pair') 
        plt.title('100 day and 20 day MA: '+str(self.base_currency)+'-'+ticker+' pair')
        plt.savefig('./app/static/image_price_ts_basic.png',format='png')
        return('static/image_price_ts_basic.png')
        
    def get250Day(self,ticker):
        url='https://min-api.cryptocompare.com/data/histoday'
        #will return a str
#prices from last 200 days
        parameters= {'fsym':ticker, 'tsym': self.base_currency, 'e': 'Bittrex', 'aggregate':1,'limit':450}
        
        r=requests.get(url,parameters)
        #handle this error in the calling class
        j_obj=r.json()
        if j_obj['Response']=='Error':
            print(j_obj)
            #print("fetch didn't work")
            raise RuntimeError
        raw_time=j_obj['Data']
        df=pd.DataFrame.from_dict(raw_time)
        df['time']=df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        df.index=pd.to_datetime(df.time)
        return(df)