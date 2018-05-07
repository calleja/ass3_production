#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Account class...
This object will have attributes like cash and positions and will need to allow for retrieval methods and some minor calculations of each (unless it can be done in the trade/transaction class). This object must persist throughout the trading session.

Positions must store total shares, average price, possibly VWAP
In addition, store cash amount; keep in mind that short positions do not affect cash balance.
"""
import pandas as pd
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt

class Account:
    def __init__(self):
        self.coin_bal=1000000
        self.coin_bal_history=pd.Series()
        self.pl_history=pd.Series()
        #will be a nested dictionary, the outermost key is the ticker and the value will be a dictionary of total shares, average price and possibly VWAP, realized p/l... ex: {ticker:{'notional':notaionValue,'direction':direction_string, etc...}}
        #TODO this cannot be a predefined dictionary... needs to be transformed to an empty dynamic dict
        ''' legacy
        self.positions={'AAPL':{'shares':0,'vwap':0,'realized_pl':0,'notional':0,'original_direction':'','upl':0},
                        'INTC':{'shares':0,'vwap':0,'realized_pl':0,'notional':0,'original_direction':'','upl':0},
                        'MSFT':{'shares':0,'vwap':0,'realized_pl':0,'notional':0,'original_direction':'','upl':0},
                        'SNAP':{'shares':0,'vwap':0,'realized_pl':0,'notional':0,'original_direction':'','upl':0},
                        'AMZN':{'shares':0,'vwap':0,'realized_pl':0,'notional':0,'original_direction':'','upl':0}}
        '''
        self.positions={}
        
        
    def getCash(self):
        print('coin balance is :'+str(self.coin_bal))
        return self.coin_bal
    
    def getPortfolio(self):
        return(self.positions)
    
        
    def checkIfNew(self,dic):
        #checks whether there is a position in that stock and in the same direction
        print('the ticker stored in the ass1 argument is {}'.format(dic['ticker']))
        try:
            if dic['ticker'] in self.positions.keys():
                if self.positions[dic['ticker']]['original_direction']==dic['original_tradetype']:
                    return False
            else:
            #print('entering this new position into the accounts dictionary - from the ass1_accountsClass')
            #create an entry/holding in the portfolio for that stock at 0 notional and shares
                self.positions[dic['ticker']]={'coins':0,'notional':0,'original_direction':'','realized_pl':0}
            #print('trade attributes stored:\n')
            #print(self.positions[dic['ticker']])
                return True
        except TypeError:
            print('The type error was thrown at accounts_class. Trying to call the function again.')
            self.positions={}
            self.checkIfNew(self,dic)
        
        
    def postEquityTrade(self,dic):
        #dic will come from the tradeClass
        '''
  the dictionary will contain total number of shares and trade price... conditional statements will qualify whether the trade serves to: 
    a) open a new position - can be long or short
    b) close all or part of an existing position - long or short
    c) augment an existing position - short or long

this function will then instantiate a tradeClass object that will QA the trade (verify whether legal), then subsequently amend the current portfolio
'''
        #if the trade/position is new, create a new entry in the dictionary to not trigger errors
        isNew=self.checkIfNew(dic) #if the trade is in a new ticker, than no need to calculate realized pl
        if isNew:
            self.positions[dic['ticker']]['vwap']=dic['notional_delta']/dic['position_delta']
        else: #pre-existing position in the stock
            self.calcVWAP(dic) #function will determine whether VWAP is to be impacted, and if so, will update it
            self.calcRealizedPL(dic) #calclates realized PL if applicable
            #calculate the number of shares in portfolio
        self.positions[dic['ticker']]['coins']=dic['position_delta']+self.positions[dic['ticker']]['coins']    
        #calculate notional of shares held
        self.positions[dic['ticker']]['notional']=self.positions[dic['ticker']]['coins']*self.positions[dic['ticker']]['vwap']
        #calculate the cash position after the trade
        self.coin_bal=dic['cash_delta']+self.coin_bal
        
        #tracking the original_tradetype
        
        #clean up the portfolio if there are no positions
        if self.positions[dic['ticker']]['coins']==0:
            self.positions[dic['ticker']]['original_direction']=''
            self.positions[dic['ticker']]['vwap']=0
        else:
            self.positions[dic['ticker']]['original_direction']=dic['original_tradetype']
        #updat self.positions[ticker]['realized_pl'] with another application
        #print('portfolio after the conclusion of postEquityTrade in the accounts class')
        #print(self.positions)

    def calcVWAP(self,dic):
        #reconcile the new transaction to the previous portfolio stats... only applicable to position increasing transactions (buys for long positions and sales for shorts)
        #ticker will be in the dictionary... no need to check that... initial trade will not have a VWAP
        #print("I'm reconciling the portfolio to this transaction dictionary:")
        #print(dic)
        if(self.positions[dic['ticker']]['original_direction']==dic['original_tradetype'] and  abs(self.positions[dic['ticker']]['coins']+dic['position_delta'])>abs(self.positions[dic['ticker']]['coins'])):
            newVWAP=(self.positions[dic['ticker']]['notional']+dic['notional_delta'])/(self.positions[dic['ticker']]['coins']+dic['position_delta'])
            self.positions[dic['ticker']]['vwap']=newVWAP
        elif (self.positions[dic['ticker']]['vwap']==0 and  abs(self.positions[dic['ticker']]['coins']+dic['position_delta'])>=abs(self.positions[dic['ticker']]['coins'])):
            newVWAP=(self.positions[dic['ticker']]['notional']+dic['notional_delta'])/(self.positions[dic['ticker']]['coins']+dic['position_delta'])
            self.positions[dic['ticker']]['vwap']=newVWAP    
        else:
            return
        
    def calcRealizedPL(self,dic):
        #requires a calculated VWAP... realized PL = notional on transaction - VWAP * same # of shares, so price is not necessary
        if(self.positions[dic['ticker']]['original_direction']==dic['original_tradetype'] and abs(self.positions[dic['ticker']]['coins']+dic['position_delta'])<self.positions[dic['ticker']]['coins']):
            #'notional_delta' is negative for sales
            self.positions[dic['ticker']]['realized_pl']=-dic['notional_delta']+self.positions[dic['ticker']]['vwap']*dic['position_delta']+self.positions[dic['ticker']]['realized_pl']
            
    def calcUPL(self,dictOfPrices,sorted_list):
        #dictOfPrices = output from scrape class; format: {ticker as str:price as float}
        #calc = portfolio for >0 holdings: current market price*shares held - VWAP*shares held  
        #original version of this function sorts the p/l table by trade date... a sorted list of trade tickers was the second parameter of this function... this has been removed
        total_notional=0
        
        #iterate through the positions dictionary
        for k,v in self.positions.items():
            #print('calculating upl for {}'.format(k))
            #TODO retrieve price... current version won't work: dictOfPrices is a list of dictionaries, not a straight dictionary
            self.positions[k]['upl']=dictOfPrices[k]['Bid']*v['coins']-v['vwap']*v['coins']
            g=dictOfPrices[k]['Bid']*v['coins']
            self.positions[k]['notional']=g
            total_notional+=g 
            #new spec for ass2: sum the UPL and RPL for each position
            self.positions[k]['total p/l']=self.positions[k]['upl']+self.positions[k]['realized_pl']
            #TODO call a separate function that calculates the total # of shares and calculates the proportion of each component holding... better served if this is done AFTER the positions dict is converted to a pd.DataFrame
        #print('p/l calcs now complete{} now calculating cash portion'.format(self.positions))
         
         #calculate the total size of portfolio: cash + notional
        self.portfolio_value=self.coin_bal+total_notional
        cash_line={'cash':{'coins':self.coin_bal,'notional':self.portfolio_value}}
        self.coin_bal_history=        self.coin_bal_history.append(pd.Series(self.coin_bal,index=[datetime.datetime.now()]))
        #TODO cash_line will need to conform to the table structure: with index "cash" and blank values for WAP, UPL and RPL... ensure that RPL persists after the position in the stock was liquidated
        #print('cash is now calculated as {}'.format(cash_line))
        #print('now calling conver2Df from the accounts class')
        
        #convert portfolio to pd.DataFrame and append the calculated cash_line
        return(self.convert2Df(cash_line,sorted_list))
    
    
    def convert2Df(self,cash_line,sort_list):
        #index of the df should be the coin symbols
        df=pd.DataFrame.from_dict(self.positions,orient='index')   
        df=df.loc[df['coins'].notnull(),:]
        #print('DF of positions now set in conver2DF, having shape {}'.format(df.shape))
        #print('df is of type {}'.format(type(df)))
        
        ''' the code died somewhere here '''
        #print('below is the first of the series of functions that died... a sort function... it makes use of sorted list{}... and the dataframe to which it is applied is{}'.format(sort_list,df))
        
        try:
            #df.set_index('coins',inplace=True,drop=False)
            #print('sorting by the following list {}'.format(sort_list))
            #print('the index of the portfolio df {}'.format(df.index.values))
            #print('the sorted list has {} number of elements.'.format(len(sort_list)))
            df1=df.loc[sort_list.tolist(),:].copy()
        except KeyError as e:
            print('this error coming from accounts class, specifically{}'.format(sys.exc_info()))
            print(e)
            print(e.with_traceback)
            print(e.__traceback__)
            #in order to proceed w/o sorting
            df1=df.copy()
            #print('df post sorting {}'.format(df1))
        df1['proportion_shares']=df1.apply(lambda x: x['coins']/  sum(df['coins']),axis=1)
        #print('df after calculating proportion shares{}'.format(df1))
        
        df1['proportion_notional']=df1.apply(lambda x: x['notional']/sum(df1['notional']),axis=1)
        #print('df after calculating proportion notional {}'.format(df1))
        total_pl=df1['total p/l'].sum(skipna=True)
        self.pl_history=        self.pl_history.append(pd.Series(total_pl,index=[datetime.datetime.now()]))
        #compile the cash dataframe
        cash_line['cash'].update({'original_direction':'','realized_pl':'','vwap':0,'total p/l':None,'proportion_shares':None,'proportion_notional':None,'upl':None})
        #print('cash line of the df is now ready for df conversion {}'.format(cash_line))
        
        cash_df=pd.DataFrame.from_dict(cash_line,orient='index')
        
        ''' compiler did not reach here '''
        
        #print('cash df is set as {}'.format(cash_df))
        #print('portfolio df is set as {} '.format(df1))
        #print('and the result of EVERYTHING is {}'.format(pd.concat([df1,cash_df],axis=0,ignore_index=False)))
        self.runGraphs()
        #return a df and a string
        return(pd.concat([df1,cash_df],axis=0,ignore_index=False))
    
    def runGraphs(self):
        fig,(ax1,ax2)=plt.subplots(1,2)
        ax1.plot(self.coin_bal_history)
        ax1.tick_params(labelrotation=45)
        ax1.set_title('Coin Balance History')
        ax2.plot(self.pl_history)
        ax2.tick_params(labelrotation=45)
        ax2.set_title('Portfolio P/L History')
        plt.savefig('./app/static/portfolio_level.png',format='png')
        
'''
fig,(ax1,ax2)=plt.subplots(1,2)
        ax1.plot(df_mongo['execution timestamp'],df_mongo['executed price'])
        ax1.tick_params(labelrotation=45)
#plt.xticks(rotation=45)
        ax2.plot(df_mongo['execution timestamp'],df_mongo['vwap'])
        ax2.tick_params(labelrotation=45)
        plt.savefig('./app/static/image_history.png',format='png')
'''        
        
