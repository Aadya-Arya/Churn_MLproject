import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("splitting training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            
            
            models={
                "Logistic Regression":LogisticRegression(),
                "K-Neighbors Classifier":KNeighborsClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Random Forest Classifier":RandomForestClassifier(),
                "XGB Classifier": XGBClassifier(),
                "Gradient Booster":GradientBoostingClassifier(),
                "CatBoosting":CatBoostClassifier(verbose=False),
            }
            params={
                "Logistic Regression":{
                    'C':[0.01,0.1,1,10,100],
                    'penalty':['l1','l2'],
                    'solver':['liblinear']},
                "K-Neighbors Classifier":{
                    'n_neighbors':[3,5,7,9,11],
                    'weights':['uniform','distance'],
                    'metric':['euclidean','manhattan']},
                "Decision Tree":{
                    'max_depth':[3,5,7,10,None],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,4],
                    'criterion':['gini','entropy']},
                "Random Forest Classifier":{
                     'n_estimators':[100,200],
                     'max_depth':[10,None],
                     'min_samples_split':[2,5],
                     'min_samples_leaf':[1,2],
                     'max_features':['sqrt']},
                "XGB Classifier":{
                    'n_estimators':[100,200,300],
                    'learning_rate':[0.01,0.05,0.1],
                    'max_depth':[3,5,7],
                    'subsample':[0.8,1.0],
                    'colsample_bytree':[0.8,1.0]},
                "Gradient Booster":{
                    'n_estimators':[100,200],
                    'learning_rate':[0.01,0.05,0.1],
                    'max_depth':[3,5]},
                "CatBoosting":{
                    'iterations':[100,200,500],
                    'learning_rate':[0.01,0.05,0.1],
                    'depth':[4,6,8],
                    'l2_leaf_reg':[1,3,5]},
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            best_model_score=max(model_report.values())
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("no best model found",sys)
            logging.info("found the best model for the training and testing dataset")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted=best_model.predict(X_test)
            accuracy=accuracy_score(y_test,predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)