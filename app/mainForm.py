from flask_wtf import FlaskForm
from wtforms import RadioField, BooleanField, SubmitField, StringField
from wtforms.validators import DataRequired

''' This contains all the elements of the web form, and will connect to the login.html protocol saved in the tamplates folder '''

class MainMenuForm(FlaskForm):
#notice that we are extending FlaskForm in the above class constructor

#the below are subclasses with their own attributes, which we invoke in the login.html template
    menu_selection=RadioField('What would you like to do', choices=[('a','Trade'),('b','Show Blotter'),('c','Show P/L'),('d','Show Coin Level Details')])
    #str_selection=StringField('Activity', validators=[DataRequired])
    submit = SubmitField('Submit')