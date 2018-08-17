from flask_wtf import FlaskForm
from wtforms import RadioField, BooleanField, SubmitField, StringField, validators
from wtforms.validators import DataRequired

''' This contains all the elements of the web form, and will connect to the login.html protocol saved in the tamplates folder '''

class MainMenuForm(FlaskForm):
#notice that we are extending FlaskForm in the above class constructor

    boro=StringField('Boro')
    block=StringField('Block')
    lot=StringField('Lot')
    submit=SubmitField('Submit')