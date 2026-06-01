import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def binary_mapping(self,X):
        mapping={'Yes':1,'No':0,'Female':1,'Male':0}
        X=X.copy()
        for col in X.columns:
            X[col]=X[col].map(mapping)
        return X
    
    def frequency_encoder(self,X,freq_map):
        X=X.copy()
        for col in X.columns:
            X[col]=X[col].map(freq_map).fillna(0)
        return X


    def get_data_tramsformer_object(self, city_frequency):
        '''
        This function is for transforming the data.
        '''
        try:
            binary_cols=['Gender','Senior Citizen','Partner','Dependents','Phone Service','Paperless Billing']
            ordinal_cols=['Contract']
            high_cardinality_cols = ['City']
            numerical_col=['Tenure Months', 'Monthly Charges','Total Charges']
            nominal_col=['Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Payment Method']

            numerical_pipeline=Pipeline(steps=[("scaler",StandardScaler())])
            ordinal_pipeline=Pipeline(steps=[('ordinal',OrdinalEncoder(categories=[['Month-to-month','One year','Two year']]))])
            nominal_pipeline=Pipeline(steps=[("Nominal",OneHotEncoder())])
            binary_pipeline=Pipeline(steps=[('Binary',FunctionTransformer(self.binary_mapping))])
            city_pipeline=Pipeline(steps=[('frequency',FunctionTransformer(self.frequency_encoder,kw_args={'freq_map':city_frequency}))])

            preprocessor=ColumnTransformer(transformers=[
                ('numerical_pipeline',numerical_pipeline,numerical_col),
                ('nominal',nominal_pipeline,nominal_col),
                ('ordinal',ordinal_pipeline,ordinal_cols),
                ('city',city_pipeline,high_cardinality_cols),
                ('binary',binary_pipeline,binary_cols)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            logging.info("Reading the train and test data successfully ")
            train_data['Total Charges']=pd.to_numeric(train_data['Total Charges'],errors='coerce')
            test_data['Total Charges']=pd.to_numeric(test_data['Total Charges'],errors='coerce')
            train_data.dropna(subset=['Total Charges'],inplace=True)
            train_data.dropna(subset=['Total Charges'],inplace=True)
            logging.info("Done with the data type issue in Total Charges")

            drop_cols=['CustomerID','Count','Country','State','Lat Long','Churn Label','Churn Score','Churn Reason','CLTV','Zip Code', 'Latitude', 'Longitude']
            train_data.drop(columns=drop_cols,inplace=True)
            test_data.drop(columns=drop_cols,inplace=True)

            logging.info("The unnecessary columns are dropeed from training and testing data")
            city_frequency=(train_data['City'].value_counts().to_dict())
            
            target_variable='Churn Value'
            X_train=train_data.drop(columns=target_variable)
            X_test = test_data.drop(columns=target_variable)
            y_train=train_data[target_variable]
            y_test=test_data[target_variable]
            logging.info("Splitting the data into training and testing data is done")

            preprocessing_obj=self.get_data_tramsformer_object(city_frequency)
            input_feature_train_arr=preprocessing_obj.fit_transform(X_train)
            input_feature_test_arr=preprocessing_obj.fit_transform(X_test)
            logging.info("Preprocessing completed")

            train_arr=np.c_[input_feature_train_arr,np.array(y_train)]
            test_arr=np.c_[input_feature_test_arr,np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            logging.info("Preprocessor pickle file is saved")

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)

            



    