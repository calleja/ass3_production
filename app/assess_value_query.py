# -*- coding: utf-8 -*-
"""
AV query for a bbl
"""

import pymongo
import pandas as pd
import numpy as np


def lookupAV(collection,bbl):
    result=collection.find({'_id':bbl},{'AV':1,'_id':0})
    gh=list(result)
    df=pd.DataFrame.from_dict(gh[0]['AV'],orient='columns')
    return(df.to_html())
    
    '''result=filings.find({'_id':1000070028},{'AV':1,'_id':0})

gh=list(result)
df=pd.DataFrame.from_dict(gh[0]['AV'],orient='columns')

df.to_html() '''

