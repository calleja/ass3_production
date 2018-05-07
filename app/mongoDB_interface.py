#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:47:44 2018

@author: lechuza
"""

import pymongo
import pandas as pd
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

class MongoInterface(object):
    def __init__(self):
        self.checkConnection()

    def checkConnection(self):
        try:
            client=pymongo.MongoClient('mongodb://tio:persyy@ds131329.mlab.com:31329/crypto_trades',serverSelectionTimeoutMS=3000)
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as err:
            #try to connect one more time
            print('trying to connect one more time')
            client=pymongo.MongoClient('mongodb://tio:persyy@ds131329.mlab.com:31329/crypto_trades',serverSelectionTimeoutMS=7000)
        self.db=client.crypto_trades
        self.trades=self.db.trade_collection
        self.trades.remove() 
        
    def tradeInjection(self,doc):
        self.trades.insert_one(doc)
        return
    
    def retrieveTrades(self):
        try:
            rs=self.trades.find().sort('execution timestamp',pymongo.DESCENDING)
        #.sort('field', pymongo.ASCENDING)
        #works in robo3t: db.trade_collection.find().sort({'execution timestamp':-1})
            df=pd.DataFrame(list(rs))
            print('mongodb has the following fields - retrieved from mongoDB interface - retrieveTrades(): {}'.format(df.columns))
            df.sort_values(by=['execution timestamp'], ascending=False,inplace=True)
        #print('trade blotter as retrieved from mongo {}'.format(df[['ticker','execution timestamp']]))
            return(df)
        except KeyError:
            dic={'trade price':{'asset one':None},'notional value':{'asset one':None}}
            df=pd.DataFrame.from_dict(dic)
            return(df)
    
    def clearCollection(self):
        self.trades.remove()
        return
    
    def retrieveAll(self):
        rs=self.trades.find()
        return(pd.DataFrame(list(rs)))
        
    def retrieveCoinSpecific(self,ticker):
        results=self.trades.find({'ticker':ticker},{'ticker':1,'execution timestamp':1,'executed price':1,'vwap':1,'_id':0})
        #run a query to select all documents of a particular ticker, then graph the vwap history and trade execution history
        #query resource: https://docs.mongodb.com/manual/tutorial/query-documents/
        df_mongo=pd.DataFrame(list(results))
        print('columns in the dataframe: {}'.format(df_mongo.columns))
        df_mongo.sort_values('execution timestamp')    
        print('these are the prices stored: {}'.format(df_mongo['executed price'].values))
        plt.close()
        fig,(ax1,ax2)=plt.subplots(1,2)
        ax1.plot(df_mongo['execution timestamp'],df_mongo['executed price'])
        ax1.tick_params(labelrotation=45)
        ax1.set_title('Executed Price History')
#plt.xticks(rotation=45)
        ax2.plot(df_mongo['execution timestamp'],df_mongo['vwap'])
        ax2.tick_params(labelrotation=45)
        ax2.set_title('VWAP History')
        plt.savefig('./app/static/image_history.png',format='png')
        #return the paths of two graphics: a vwap plot and trade price plot
        #db.trade_collection.find({'ticker':'LTC'},{'ticker':1,'execution timestamp':1,'executed price':1,'vwap':1,'_id':0})
        #return('static/image_history.png')
        return('image_history.png')
'''
        trade_list=[]
for i in range(5):
    trade_list.append({'notional_delta': 5.6895, 'cash_delta': -5.6895, 'position_delta': 100.0, 'ticker': 'ETH', 'original_tradetype': 'long','tradetime':datetime.datetime.now()})
    sleep(2)
    
for i in trade_list:    
    trades.insert_one(i)
'''

