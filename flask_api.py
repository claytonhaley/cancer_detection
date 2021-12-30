"""
Created on Mon Dec 20

@author Clayton Haley
"""

from flask import Flask, request, render_template
import numpy as np
import pickle
# import flasgger
# from flasgger import Swagger


app = Flask(__name__)
# Swagger(app)

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict_breast_cancer():

    # """ Let's classifiy whether a patient has breast cancer or not
    # This is using docstrings for specifications.
    # ---
    # parameters:
    #   - name: mean_radius
    #     in: query
    #     type: number
    #     required: true
    #   - name: mean_texture
    #     in: query
    #     type: number
    #     required: true
    #   - name: mean_perimeter
    #     in: query
    #     type: number
    #     required: true
    #   - name: mean_area
    #     in: query
    #     type: number
    #     required: true
    #   - name: mean_smoothness
    #     in: query
    #     type: number
    #     required: true
    # responses:
    #     200:
    #         description: Success
    # """
    mean_radius = request.args.get('mean_radius')
    mean_texture = request.args.get('mean_texture')
    mean_perimeter = request.args.get('mean_perimeter')
    mean_area = request.args.get('mean_area')
    mean_smoothness = request.args.get('mean_smoothness')
    prediction = classifier.predict([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    
    if str(prediction) == "[0]":
        result = "The patient does not have cancer."
        return render_template('home.html',pred='Expected diagnosis will be {}'.format(result))
    result = "The patient has cancer."
    return render_template('home.html', pred='Expected diagnosis will be {}'.format(result))

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=8080, debug=1)