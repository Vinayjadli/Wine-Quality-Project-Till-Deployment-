# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:43:57 2020

@author: user
"""
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in= open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome all'

@app.route('/predict',methods=["Get"])
def prdict_house_price():
    """Let's predict the wine quality 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: type
        in: query
        type: number
        required: true
      - name: fixedacidity
        in: query
        type: number
        required: true
      - name: volatileacidity
        in: query
        type: number
        required: true
      - name: citricacid
        in: query
        type: number
        required: true
      - name: residualsugar
        in: query
        type: number
        required: true
      - name: chlorides
        in: query
        type: number
        required: true
      - name: freesulfurdioxide
        in: query
        type: number
        required: true
      - name: totalsulfurdioxide
        in: query
        type: number
        required: true
      - name: density
        in: query
        type: number
        required: true
      - name: pH
        in: query
        type: number
        required: true
      - name: sulphates
        in: query
        type: number
        required: true
      - name: alcohol
        in: query
        type: number
        required: true
  
    responses:
        200:
            description: The output values
    """        
    
    type=request.args.get('type')
    fixedacidity=request.args.get('fixedacidity')
    volatileacidity=request.args.get('volatileacidity')
    citricacid=request.args.get('citricacid')
    residualsugar=request.args.get('residualsugar')
    chlorides=request.args.get('chlorides')
    freesulfurdioxide=request.args.get('freesulfurdioxide')
    totalsulfurdioxide=request.args.get('totalsulfurdioxide')
    density=request.args.get('density')
    pH=request.args.get('pH')
    sulphates=request.args.get('sulphates')
    alcohol=request.args.get('alcohol')

    prediction=classifier.predict([[type, fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol]])

    return "The predicted Values is"+ str(prediction)


@app.route('/predict_file' ,methods=["post"])
def prdict_price():
    """Let's predict the wine quality 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
    """
    
    df_test= pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted Values of the csv is"+ str(list(prediction))



if __name__ == '__main__':
    app.run()