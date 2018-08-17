# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:15:20 2018

@author: callejal
"""

from flask_wtf import FlaskForm
from wtforms import RadioField, BooleanField, SubmitField, StringField
from wtforms.validators import DataRequired


class ConfirmForm(FlaskForm):
    submit=SubmitField('OK')
        
	