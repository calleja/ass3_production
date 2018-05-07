#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client dialogue
"""
import sys
from app import tradeClass as trade
from app import ass1_acountsClass as accts
from app import tradeManager as tm
from app import retrieveMarkets as rm
from app import mongoDB_interface as mongo
import datetime

class Dialogue(object):
    def __init__(self):
        #create a new account/portfolio
        self.act=accts.Account()
        self.mongo_connection=mongo.MongoInterface()
        self.todayTrading=tm.TradingDay(self.mongo_connection)
    
    def engageUser(self, menuSelection, ticker=None,qty=None ,tradetype=None, confirmed=False):
        #menuSelection=input('Please select from the list of options below.\n a -Trade\n b - Show Blotter\n c- Show P/L\n d - Quit\n > ')
        if menuSelection=='a':
            #if 'a' and they only want to view markets, return a graphic and the current price; book the trade if they want to submit a trade
            print('sending the following over to prepareTrade: {}, {}, {}, {}'.format(ticker,qty,tradetype,confirmed))
            #returns two items: the price string and path string of the graphic... def prepareTrade(self,ticker,qty,tradetype,confirmed):
            return(self.prepareTrade(ticker=ticker,qty=qty,tradetype=tradetype,confirmed=confirmed))
        elif menuSelection=='b':
            #call the blotter from the tradeManager class - may need rendering in this class, and the return value from either this function or another in this class can be handled at the controller level
            # TODO print('call blotter function')
            #the blotter function will return a list of dictionaries, or perhaps a pandas dataframe, that I'll then print... if extensive formating is required, I'll do it in this class
            return(self.todayTrading.prettyPrintTradeLog().iloc[:,1:].to_html(index=False))
        elif menuSelection=='c':
            #show p/l
            print('your current portfolio is below... p&l calc is pending')
            return(self.calcPL())
            
        elif menuSelection=='d':
            return(self.mongo_connection.retrieveCoinSpecific(ticker))
        else:
            return(print('please select an option'))

    
    def iterateDF(self,df,index_start):
        #begin iteration from the index passed in the argument call
        g=index_start
        increment=35
        print(df.loc[:,['Currency','CurrencyLong']].iloc[g:g+increment])
        g=g+increment
        user_input=input('which coin would you like to trade? Please type the index number corresponding with the coin symbol.\n> ')
        if user_input=='n':
            while user_input=='n' and   g<df.shape[0]+(increment-2):
                print(df.loc[:,['Currency','CurrencyLong']].iloc[g:g+increment])
                g=g+increment
                user_input=input('> ')
            #TODO user_input ne 'n' breaks the loop, must then evaluate the user input
            #TODO do I need to return the series, or ticker, or nothing?
        return(user_input,g)
    
    def qaAndSelect(self,shapes,userInput):
        #evaluate user input
        try:
            lookup_index=int(userInput)
            if (lookup_index < shapes and int(lookup_index)>-1):
                return(True)
            else:
                return(False)
        except ValueError:
            #user did not enter an appropriate value
            print("please type an integer or the letter 'n'")
            return(False)
    
    def prepareTrade(self,ticker,qty,tradetype,confirmed):
        #key word arguments will contain the ticker
        #aggDic will be the dictionary that stores trade details and later passes to fa
        self.rm=rm.RetrieveMarkets()
        agg_dic={}            
            #dictionary of trade stats to send over to the tradeClass
                    #AFTER TICKER IS SELECTED
        agg_dic['ticker']=ticker
        print('the ticker stored in the agg_dic in the prepareTrade() is {}'.format(agg_dic['ticker']))
        options={'a':'buy','b':'sell to close'}
        agg_dic['tradetype'],agg_dic['price']=options[tradetype],self.selectExecPrice(letter=tradetype,ticker=ticker)
        if confirmed==False:
                #return the current price and the stock chart
            url=self.rm.get100Day(ticker)
            #print('engageUser is sending back price: {}'.format(agg_dic['price']))
            #print('eu.prepareTrade url is sending back price: {}'.format(url))
            #returns the price as a string and a path as string for location of the price chart
            return(str(agg_dic['price']),str(url))
               # '.format(agg_dic['price'],self.rm.base_currency))
        else:
                #record the trade
            agg_dic['coins']=qty    
            agg_dic['timestamp']=datetime.datetime.now()
                #TODO record trade details and verify validity (by interacting with the account class)... recall that it's the tradeManager that will store/send the trades to the mongoDB
            try:
                #todayTrading.makeTrade() returns nothing
                self.todayTrading.makeTrade(agg_dic,self.act)
                    #print(single_trade_dic)
                    #direct interface with accounts class
                print('assume we have logged your trade')
                    #print(self.act.positions)
            except KeyError:
                print('try a valid trade')

                
    def selectExecPrice(self,letter,ticker):
        #handle the user trade request and lookup the proper price
        try:
                #getCurrentPrice() requires a list of tickers - fine if contains one element
            current_price_float=self.rm.getCurrentPriceCC([ticker])['BTC']
            #select the appropriate price according to the trade type: buy on ask and sell on the bid
            appPrice=current_price_float
            return(appPrice)
        except ValueError:
            print('type either a or b')
            raise ValueError
            
            
    def calcPL(self):
        #call scraper, pass dictionary of current prices to the account object and print the current status of the portfolio dictionary, equipped with both realized and unrealized p+l
        #ideally, get all the tickers from the accounts class, then pass an array to retrieveMarkets class
        #this should be coin tickers
        #print('acquiring keys from the accounts portfolio df')
        #print(self.act.positions.keys())
        #keys function returns a list
        ticker_array=self.act.positions.keys()
        if len(ticker_array)>0:
            
        #verified this works independent of the program
            prices_dict=self.rm.getCurrentPrice(ticker_array)
        
        
        #TODO sortedTrades relies on the old tuple format and not the new mongoDB... sortTrades() returns a 
        #TODO call the tradeManager class to retrieve the trade blotter and return a unique list of tickers in order of trade activity
        #TODO ensure that set() accepts a pd.Series... has the benefit of storing only unique values... even then, need to ensure that the latest entry of a given ticker is preserved.
            #print('acquiring ordered portfolio')
            sorted_list=self.todayTrading.prettyPrintTradeLog().sort_values(['ticker'])['ticker'].unique()
            #print('sorted potfolio is {}'.format(sorted_list))
            #print('moving on the last step: act.calcUPL - called from eu.calcPL')
        
        #TODO undo print(self.act.calcUPL(prices_dict,sorted_list))
            return(self.act.calcUPL(prices_dict,sorted_list).to_html())
        else:
            return('<p> Your cash balance is {}</p>'.format(self.act.coin_bal))
        return
    
    def retrievePortfolio(self):
        return(self.act.positions.keys())
        #return(['BTC','DGB'])