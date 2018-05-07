#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:37:04 2018

@author: lechuza
"""

from flask_wtf import FlaskForm
from wtforms import RadioField, BooleanField, SubmitField, StringField
from wtforms.validators import DataRequired

''' This contains all the elements of the web form, and will connect to the login.html protocol saved in the tamplates folder '''

class CryptoMenuForm(FlaskForm):
#notice that we are extending FlaskForm in the above class constructor

#the below are subclasses with their own attributes, which we invoke in the login.html template
    str_selection=StringField('Index Number')
    nexty = SubmitField('Next')
    submit = SubmitField('Submit')