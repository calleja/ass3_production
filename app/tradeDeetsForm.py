# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:35:05 2018

@author: CallejaL
"""

from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, StringField
from wtforms.validators import DataRequired

''' This contains all the elements of the web form, and will connect to the login.html protocol saved in the tamplates folder '''

class TradeDetailsForm(FlaskForm):
#notice that we are extending FlaskForm in the above class constructor

#the below are subclasses with their own attributes, which we invoke in the login.html template
    #a- buy\n b- sell to close\n
    #menu_selection=RadioField('What would you like to do', choices=[('a','Trade'),('b','Show Blotter'),('c','Show P/L')])
    tradeRadio=RadioField('Select from the trade types',choices=[('a','buy'),('b','sell to close')])
    coinFreqString=StringField('Number of coins to trade', validators=[DataRequired()])
    viewstats=SubmitField('View Stats')
    submit = SubmitField('Submit')
