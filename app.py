import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('SVM.sav', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET','post'])
def predict():
	
	Gender = int(request.form['Gender'])
	SSC_P = float(request.form['SSC_P'])
	SSC_B = int(request.form['SSC_B'])
	HSC_P = float(request.form['HSC_P'])
	HSC_B = int(request.form['HSC_B'])
	DEGREE_P = float(request.form['DEGREE_P'])
	WORK_EX = int(request.form['WORK_EX'])
	EXAM_SCORE = float(request.form['EXAM_SCORE'])
	SPECIALISATION = int(request.form['SPECIALISATION'])
	MBA_PERCENTAGE = float(request.form['MBA_PERCENTAGE'])
	ARTS = int(request.form['ARTS'])
	COMMERCE = int(request.form['COMMERCE'])
	SCIENCE = int(request.form['SCIENCE'])
	COMMERCE_MANAGEMENT= int(request.form['COMMERCE_MANAGEMENT'])
	OTHER_DEGREE = int(request.form['OTHER_DEGREE'])
	SCIENCE_TECHNOLOGY = int(request.form['SCIENCE_TECHNOLOGY'])
	
	final_features = np.array([Gender ,SSC_P ,SSC_B  ,HSC_P  ,HSC_B  ,DEGREE_P  ,WORK_EX  ,EXAM_SCORE  ,SPECIALISATION  ,MBA_PERCENTAGE   ,ARTS  ,COMMERCE  ,SCIENCE  ,COMMERCE_MANAGEMENT,OTHER_DEGREE  ,SCIENCE_TECHNOLOGY])
	final_features = final_features.reshape(1,-1)
	predict = model.predict(final_features)
	
	output = predict[0]
	if(output) :
		PREDICTION ="Placed"
	else :
		PREDICTION = "Not Placed"
	
	return render_template('index.html', prediction_text='model predicted you will be {}'.format(PREDICTION ))
	
if __name__ == "__main__":
	app.run(debug=True)
