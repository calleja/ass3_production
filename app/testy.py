# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:33:37 2018

@author: CallejaL
"""
#test engageUser functions
import sys
sys.path.append('C:/Users/callejal/Documents/flask_work-master/flask_work-master/crypto-test-env')
from app import engageUser as eu
from app import retrieveMarkets as rm
import os

eud=eu.Dialogue()
eud.engageUser(menuSelection='a',ticker='XMR',tradetype='a')

mark=rm.RetrieveMarkets()
mark.getCurrentPriceCC(['XMR'])



g={'ETH': 13.65}
for i,k in g.items():
    print('{}{}'.format(i,k))
    
pathy=os.getcwd()+'mama'


import sys 
sys.path.append('/home/lechuza/Documents/CUNY/data_607/ass3_production')
from app import forecasts
import pandas as pd

f=forecasts.Forecast()
g=f.runProgram()

df=pd.DataFrame.from_dict({j:k for j,k in g.items()},orient='index')
gh=df.applymap(lambda x:'%.6f'%(x))

gh.dtypes


predict_html=.to_html(index=False)