from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipleline import CustomData,PredictPipeline

application=Flask(__name__)
app=application
# Route for a home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata',methods =['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('input.html')
    else:
        data=CustomData(
            Gender=request.form.get("Gender"),
            City=request.form.get("City"),
            Senior_Citizen=request.form.get('Senior_Citizen'),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            Phone_Service=request.form.get('Phone_Service'),
            Paperless_Billing=request.form.get('Paperless_Billing'),
            Contract=request.form.get('Contract'),
            Multiple_Lines=request.form.get('Multiple_Lines'),
            Internet_Service=request.form.get('Internet_Service'),
            Online_Security=request.form.get('Online_Security'),
            Online_Backup=request.form.get('Online_Backup'),
            Device_Protection=request.form.get('Device_Protection'),
            Tech_Support=request.form.get('Tech_Support'),
            Streaming_TV=request.form.get('Streaming_TV'),
            Streaming_Movies=request.form.get('Streaming_Movies'),
            Payment_Method=request.form.get('Payment_Method'),
            Tenure_Months=float(request.form.get('Tenure_Months')),
            Monthly_Charges=float(request.form.get('Monthly_Charges')),
            Total_Charges=float(request.form.get('Total_Charges'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        return render_template('input.html',results=result[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)




