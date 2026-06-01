import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl' 
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,    
        Gender :str,
        City:str,
        Senior_Citizen:str,
        Partner:str,
        Dependents:str,
        Phone_Service:str,
        Paperless_Billing:str,
        Contract:str,
        Multiple_Lines:str,
        Internet_Service:str,
        Online_Security:str,
        Online_Backup:str,
        Device_Protection:str,
        Tech_Support:str,
        Streaming_TV:str,
        Streaming_Movies:str,
        Payment_Method:str,
        Tenure_Months:int,
        Monthly_Charges:int,
        Total_Charges:int):

        self.Gender=Gender
        self.City=City
        self.Senior_Citizen=Senior_Citizen
        self.Partner=Partner
        self.Dependents=Dependents
        self.Phone_Service=Phone_Service
        self.Paperless_Billing=Paperless_Billing
        self.Contract=Contract
        self.Multiple_Lines=Multiple_Lines
        self.Internet_Service=Internet_Service
        self.Online_Security=Online_Security
        self.Online_Backup=Online_Backup
        self.Device_Protection=Device_Protection
        self.Tech_Support=Tech_Support
        self.Streaming_TV=Streaming_TV
        self.Streaming_Movies=Streaming_Movies
        self.Payment_Method=Payment_Method
        self.Tenure_Months=Tenure_Months
        self.Monthly_Charges=Monthly_Charges
        self.Total_Charges=Total_Charges
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "Gender":[self.Gender],
                "City":[self.City],
                "Senior Citizen":[self.Senior_Citizen],
                "Partner":[self.Partner],
                "Dependents":[self.Dependents],
                "Phone Service":[self.Phone_Service],
                "Paperless Billing":[self.Paperless_Billing],
                "Contract":[self.Contract],
                "Multiple Lines":[self.Multiple_Lines],
                "Internet Service":[self.Internet_Service],
                "Online Security":[self.Online_Security],
                "Online Backup":[self.Online_Backup],
                "Device Protection":[self.Device_Protection],
                "Tech Support":[self.Tech_Support],
                "Streaming TV":[self.Streaming_TV],
                "Streaming Movies":[self.Streaming_Movies],
                "Payment Method":[self.Payment_Method],
                "Tenure Months":[self.Tenure_Months],
                "Monthly Charges":[self.Monthly_Charges],
                "Total Charges":[self.Total_Charges]}
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)

    


    