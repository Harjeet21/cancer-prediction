import pickle
import numpy as np
from flask import Flask, render_template,request

#variables
app=Flask(__name__)
loadedModel=pickle.load(open('KNN Model.pkl','rb'))
scaler = pickle.load(open('Scaler.pkl','rb'))

#Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    concave_points_worst = request.form['concave points_worst']
    perimeter_worst = request.form['perimeter_worst']
    radius_worst = request.form['radius_worst']
    concave_points_mean = request.form['concave points_mean']
    perimeter_mean = request.form['perimeter_mean']

    values = scaler.transform(np.array([[ concave_points_worst,perimeter_worst,radius_worst,concave_points_mean,perimeter_mean]]))
    prediction = loadedModel.predict(values)

    if prediction[0]==0:
        prediction='B means Benign = Not Cancerous'
    else:
        prediction = 'M means Malignant= Cancerous'

    return render_template('index.html', api_output=prediction)

#main function
if __name__ == '__main__':
    app.run(debug=True) 