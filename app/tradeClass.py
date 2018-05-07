#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class QAs the trade, makes calculations based on tradetype... trades are to be stored via the tradeManager class... this class is accessed by the tradeManager and not directly
"""
import sys
#these paths will later need to be edited and matched with the imported/downloaded git folder
#sys.path.append('/usr/src/app/PROJECT_FOLDER')
#sys.path.append('C:/Users/callejal/Documents/crypto_flask/library')

class EquityTrade():
    
    def __init__(self,trade_dic,acctSnapshot):
        #trade_dic is the dictionary/json object scraped from yahoo
        self.ticker=trade_dic['ticker']
        self.price=trade_dic['price']
        self.coins=trade_dic['coins']
        self.timestamp=trade_dic['timestamp']
        self.tradetype=trade_dic['tradetype'] #buy, sell, short
        #self.original_trade_type=trade_dic['original_tradetype']
        self.currentPortfolio=acctSnapshot #ass1_accountsClass
        print('equityTrade object created in tradeClass - called from tradeManager: {}, {}, {}'.format(self.ticker, self.coins, self.currentPortfolio))
    
    def qaTrade(self,result_set):
        #ensure that the trade makes sense given the current holdings in the portfolio... return a True or False... True will allow the transaction to make all the proper updates, while a False should prompt the user that the transaction is not allowed given the current holdings
        #TODO add an entry for new cash position at the portfolio level... already referencing the cash position, so now, will append to the dictionary
        if self.tradetype=='buy':
            #cash_delta is <0 for trades to buy
            print('cash_delta of trade: '+str(result_set['cash_delta']))
            print('portfolio cash position: '+str(self.currentPortfolio.coin_bal))
            if result_set['cash_delta']+self.currentPortfolio.coin_bal<0 or self.coins<0:
                return False
            else:
                return True
        if self.tradetype=='sell to close' or self.tradetype=='buy to close': #TODO need to catch attempted sales for greater than current holding... the below inequality test should catch attempted bad sales
            try:  #return the evaluation of the conditional statement below
                return abs(self.currentPortfolio.positions[self.ticker]['coins'])>=abs(self.coins)
            except KeyError:
                return False
                
    def tradeType(self):
        #call the appropriate function, determined by trade type i.e. short, long, sell from long
        print('have arrived at tradeType() of the tradeClass object. The account object has some contents: {}'.format(self.currentPortfolio.positions))
        if self.tradetype=='short':
            result_set=self.shortTrade()
            return(result_set)
        elif self.tradetype=='buy':
            print('tradeClass.tradeType is now invoked')
            result_set=self.longTrade()
            #from above, trade dictionaries are compiled before qa
            #TODO the qaTrade method will now return a dictionary IF the trade is approved... this dictionary will contain an added field for cash position of the portfolio AFTER executing the trade
            if self.qaTrade(result_set):
                return(result_set)
            else: 
                print('trade is not legal')
                #throw an error, that will be handled in tradeManager
                raise ValueError
            #don't need to send this through the qaTrade
        elif self.tradetype=='sell to close':
            result_set=self.sellToClose()
            if self.qaTrade(result_set):
                return(result_set)
            else: 
                print('trade is not legal')
                raise ValueError
                
        elif self.tradetype=='buy to close':
            result_set=self.buyToClose()
            if self.qaTrade(result_set):
                return(result_set)
            else: 
                print('trade is not legal')
                raise ValueError
            
        
    
    def shortTrade(self):
        #no drawdown of cash, update portfolio w/negative coins
        notional_delta=self.coins*self.price
        position_delta=self.coins*-1
        cash_delta=0
        
        result_set={'notional_delta':notional_delta,'cash_delta':cash_delta,'position_delta':position_delta,'ticker':self.ticker,'original_tradetype':'short'}
        return(result_set)
        
    def buyToClose(self):
        #no drawdown of cash, update portfolio w/negative coins
        notional_delta=-1*self.coins*self.price
        position_delta=self.coins*-1
        cash_delta=notional_delta
        
        result_set={'notional_delta':notional_delta,'cash_delta':cash_delta,'position_delta':position_delta,'ticker':self.ticker,'original_tradetype':'short'}
        return(result_set)
        
    def longTrade(self):
        #cash drawdown, increase number of coins, increase in notional...
        notional_delta=self.coins*self.price
        cash_delta=notional_delta*-1 #cash debit
        position_delta=self.coins
        
        result_set={'notional_delta':notional_delta,'cash_delta':cash_delta,'position_delta':position_delta,'ticker':self.ticker,'original_tradetype':'long'}
        return(result_set)
        
    def sellToClose(self):
        #decrease #of coins held, increase cash, record realized g/l... realized g/l could be outsourced to a g/l calculator
        notional_delta=-1*self.coins*self.price
        position_delta=self.coins*-1
        cash_delta=self.coins*self.price #cash increase
        
        result_set={'notional_delta':notional_delta,'cash_delta':cash_delta,'position_delta':position_delta,'ticker':self.ticker,'original_tradetype':'long'}
        return(result_set)
        
        #call the proft calc engine and calculate realized profit
    

