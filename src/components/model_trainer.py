from dataclasses import dataclass
import os
import sys
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trainded_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training ans test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
        "LinearRegression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "KNearest Neighbour":KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Decison tree":DecisionTreeRegressor(),
        "XGBoost":XGBRegressor(),
        "CatBoost":CatBoostRegressor(verbose=False),
        "AdaBoost":AdaBoostRegressor(),
                  }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info(f'best found model on both trainng and testing dataset')
            save_object(
                file_path=self.model_trainer_config.trainded_model_file_path,
                obj=best_model
            )

            pridicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,pridicted)
            return r2_square


               
        except Exception as e:
            raise CustomException(e,sys) 

        
    
 
