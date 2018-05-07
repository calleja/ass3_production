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
