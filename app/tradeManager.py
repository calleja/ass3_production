#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class interacts with the tradeClass and stores a log of all the trades. This facilitates p+l calculation and the trade log.

Blotter requirements:
    side (Buy/Sell), Ticker
    Qty, Executed Price
    Timestamp, Money In/Out
    
P/L:
    Ticker, current position (can be 0),
    Current Market Price, VWAP,
    Unrealized p/l, realized p/l
"""
import sys
import pandas as pd
#sys.path.append('/usr/src/app/PROJECT_FOLDER')
#sys.path.append('C:/Users/callejal/Documents/crypto_flask/library')
from app import tradeClass as trade
#from app import mongoDB_interface as mongo

class TradingDay(object):
    
    def __init__(self, mongo):
        self.tradeLogTup=()
        self.mongo_connection=mongo
        
    def makeTrade(self,rawTradeDict,act):
        #TODO validate the trade, attach a "post-trade cash balance" element and record in the mongoDB db      
        #tradeClass does not interact with the account class
        print('the makeTrade method of the tradeManager class is now being called. It is processing the following dictionary: {}'.format(rawTradeDict))
        specificTrade=trade.EquityTrade(rawTradeDict,act)
        #an instantiation of EquityTrade class... no processing done yet
        
        #TODO ensure that illegal trades are not logged... this logic is contained within tradeClass.qaTrade() function
        try:
            specificTradeResult=specificTrade.tradeType()
            #we post the trade to the account class from here
            #print('sending the following to the accounts class:')
            #print(specificTradeResult)
            act.postEquityTrade(specificTradeResult)
            #retrieve the coin_bal post trade
            print('tradeManager has attempted to update the positions dictionary of the account object: {}'.format(act.getPortfolio()))
            coin_bal=act.coin_bal
            #TODO must append cash position to the trade dict... can either do here or in tradeClass... tradeClass is preferred
            formattedDic=self.prepDict(specificTradeResult,rawTradeDict,coin_bal,act)
            print('tradeManager has packaged the dictionary that will update the mongodb: {}'.format(formattedDic))
            self.mongo_connection.tradeInjection(formattedDic)
            print('your trade has been logged')
            
        except ValueError:
            #TODO ensure this breaks out of the function
            print('trade was not executed')
            raise ValueError
        #a dictionary of scrubbed and processed trade attributes... unfortunately, this does not handle an errant trade
        #TODO ensure that illegal trades are not logged... this logic is contained within tradeClass.qaTrade() function... this function does not and should not handle errant trades... need to ensure that the application breaks and that prepDic() is not called
        
        
    def prepDict(self,tradeClassDict,rawDict,coin_bal,act):        
        #take elements from both dictionaries and create a third one for logging... this will need to formatted before displaying to users
        formattedDict={'side':rawDict['tradetype'],'ticker':rawDict['ticker'],'quantity':rawDict['coins'],'executed price':rawDict['price'],'execution timestamp':rawDict['timestamp'],'money in/out':tradeClassDict['cash_delta'],'original_tradetype':tradeClassDict['original_tradetype'],'position_delta':tradeClassDict['position_delta'],'new_cash_bal':coin_bal,'vwap':act.positions[rawDict['ticker']]['vwap']}
        return(formattedDict)
        
    
    
    def prettyPrintTradeLog(self):
        #this mongo function returns a pd.DataFrame
        return(self.mongo_connection.retrieveTrades())
    
    def sortTrades(self):
        #place tickers of trades in a pandas df of ticker and timestamp; groupBy ticker and select for the latest timestamp, then sort by timestamp
        df=pd.DataFrame(list(self.tradeLogTup))

        g=df.groupby('ticker').apply(lambda x: x['execution timesestamp'].max())

        g.sort_values(ascending=False,inplace=True)
        traded_ticks=g.index.tolist()
        universe=['AMZN','AAPL','SNAP','INTC','MSFT']
        [traded_ticks.append(x) for x in universe if x not in traded_ticks]
        return(list(traded_ticks))
        
    
    
        
