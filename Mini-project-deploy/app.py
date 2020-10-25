from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
# import pickle
import pandas as pd
import joblib

app = Flask(__name__)
data = open('credit.pkl','rb')
model = joblib.load(data)

@app.route("/")
@cross_origin()
def home():
    return render_template("deploy.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":



        EDUCATION = float(request.form["Education"])
        MARRIAGE = float(request.form["Marriage-Status"])
        AGE = float(request.form["AGE"])
        LIMIT_BAL = float(request.form["LIMIT_BAL"])
        PAY_1 = float(request.form["PAY_1"])
        BILL_AMT1 = float(request.form["BILL_AMT1"])
        BILL_AMT2 = float(request.form["BILL_AMT2"])
        BILL_AMT3 = float(request.form["BILL_AMT3"])
        BILL_AMT4 = float(request.form["BILL_AMT4"])
        BILL_AMT5 = float(request.form["BILL_AMT5"])
        BILL_AMT6 = float(request.form["BILL_AMT6"])
        PAY_AMT1 = float(request.form["PAY_AMT1"])
        PAY_AMT2 = float(request.form["PAY_AMT2"])
        PAY_AMT3 = float(request.form["PAY_AMT3"])
        PAY_AMT4 = float(request.form["PAY_AMT4"])
        PAY_AMT5 = float(request.form["PAY_AMT5"])
        PAY_AMT6 = float(request.form["PAY_AMT6"])


        prediction=model.predict([[
            EDUCATION,
            MARRIAGE,
            AGE,
            LIMIT_BAL,
            PAY_1,
            BILL_AMT1,
            BILL_AMT2,
            BILL_AMT3,
            BILL_AMT4,
            BILL_AMT5,
            BILL_AMT6,
            PAY_AMT1,
            PAY_AMT2,
            PAY_AMT3,
            PAY_AMT4,
            PAY_AMT5,
            PAY_AMT6
        ]])


        output=round(prediction[0],2)

        if output == 1:
            return render_template('deploy.html',prediction_text="Credit Card holder will default-{}".format(output))
        else:
            return render_template('deploy.html',prediction_text="Credit Card holder will Not default-{}".format(output))

        # return render_template('deploy.html',prediction_text="{}".format(output))


    return render_template("deploy.html")



if __name__ == "__main__":
    app.run(debug=True)
