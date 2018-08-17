#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controller class to the equity trading platform
"""
from app import app
from flask import render_template, flash, redirect, request, session, url_for, make_response
from datetime import datetime
from app.mainForm import  MainMenuForm
from app.confirm_class import ConfirmForm
import sys
#sys.path.append('G:/Property/Luis_C/aws/rpie_app')
from app.assess_value_query import lookupAV
import pandas as pd
import numpy as np
import os
import pymongo

#set up a connection to the database, and pass it around
client = pymongo.MongoClient("mongodb://luis:persyy@127.0.0.1:27017/rpie")    
db = client.rpie
#point to the appropriate collection
filings=db['filings']

@app.route('/')
def landing(methods=['GET']):
    session['number']=0
    return render_template('landing_page.html')

bbl_dict=dict()
@app.route('/main_menu',methods=['GET','POST'])
def mainMenu():
#will need to handle the selection of the form
    form=MainMenuForm()
    if request.method == 'POST':
		#TODO don't understand redirect...
        bbl_dict['boro']=form.boro.data
        bbl_dict['block']=form.block.data
        bbl_dict['lot']=form.lot.data
        
        return(redirect('/confirm_pg'))
	#handle the rendering 
    return(render_template('main_menu.html', title='Main Menu', form=form))
	
@app.route('/confirm_pg',methods=['GET','POST'])
def confirm():    
    #can't pass a class a constructor
    form=ConfirmForm()
    bbl=bbl_dict['boro']*1000000000+bbl_dict['block']*10000+bbl_dict['lot']
    av_html=lookupAV(filings,bbl)
    if request.method == "POST":
        return(redirect('/main_menu'))
    return(render_template('confirm_page.html', title='Confirm Page', form=form,text=bbl_dict,av_table=av_html))
    
