# import flask
import sklearn

from flask import Flask, render_template, request, redirect, url_for
import joblib
import pickle

app = Flask(__name__)
loaded_model = joblib.load('/home/audumber/Desktop/BasicMLModelFlaskDeploy/model.pkl')

@app.route("/")

def root():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def make_prediction():

    if request.method == 'POST':
        exp = request.form['exp']  #taking input from user from html
        print(exp)
        X = [[float(exp)]]   #converting input into float because in dataset we have into float
        print(f'x : {X}')
        [prediction] = loaded_model.predict(X)  #it will fetch the result from loaded model (model.pkl)
        salary = round(prediction, 2)   #here will get output
    msg = "Standard salary for provided experience of  " + str(exp) + " years, would be: ₹ " + str(salary) + "/-- "

    return render_template("index.html", prediction_text= msg) #sending back to html


if __name__ == '__main__':
    app.run(debug=True)
