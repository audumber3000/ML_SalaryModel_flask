# import flask
from flask import Flask, render_template, request, redirect, url_for
from sklearn.externals import joblib
import pickle

app = Flask(__name__)


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        exp = request.form['exp']
        X = [[float(exp)]]
        [prediction] = loaded_model.predict(X)

    return render_template("predict.html", salary=round(prediction, 2), experience=exp)


if __name__ == '__main__':
    loaded_model = joblib.load('model.pkl')
    app.run(host='0.0.0.0', port=5000, debug=True)
