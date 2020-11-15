from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("Link.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Danceability = float(request.form['Danceability'])
        Energy = float(request.form['Energy'])
        Loudness = float(request.form['Loudness'])
        Speechiness = float(request.form['Speechiness'])
        Acousticness =float(request.form['Acousticness'])
        Instrumentalness = float(request.form['Instrumentalness'])
        Liveness = float(request.form['Liveness'])
        Valence = float(request.form['Valence'])
        Tempo = float(request.form['Tempo'])
        Duration_ms = float(request.form['Duration_ms'])
        pred_args = [Danceability,Energy,Loudness,Speechiness,Acousticness,Instrumentalness,Liveness,Valence,Tempo,Duration_ms]
        pred_args_arr = np.array(pred_args)
        pred_args_arr = pred_args_arr.reshape(1, -1)
        mul_reg = open("Link.pkl","rb")
        ml_model = joblib.load(mul_reg)
        model_prediction = ml_model.predict(pred_args_arr)
        model_prediction = round(float(model_prediction), 2)
        if model_prediction == 0.0:
            return render_template('index.html', prediction = "Less likely to like the song")
        else:
            return render_template('index.html', prediction = "More likely to like the song")


    return render_template('index.html', prediction = model_prediction)

if __name__ == "__main__":
    app.run(debug=True)
