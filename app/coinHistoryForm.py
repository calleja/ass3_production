# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:09:33 2018

@author: CallejaL
"""

from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired

''' This contains all the elements of the web form, and will connect to the login.html protocol saved in the tamplates folder '''

class CoinHistoryForm(FlaskForm):
    tradeRadio=RadioField('Select from the trade types')
    submit = SubmitField('Submit')
        
