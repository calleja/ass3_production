import os

#'app' is actually a global variable inside of _01_simple.py... app is an instantiation of a Flask object from the basic Flask module - as opposed to a customized one

from app import app

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=8081) #spins up a server and waits for request from client
