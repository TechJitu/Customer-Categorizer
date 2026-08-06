import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from imblearn.combine import SMOTETomek
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import SimpleImputerConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_tranasformation_config: DataTransformationConfig):
       
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_tranasformation_config
        self.data_ingestion = DataIngestion()

        self.imputer_config = SimpleImputerConfig()

        self.utils = MainUtils()
        
        
        
    
    @staticmethod
    def read_data(file_path:str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e,sys)
        
        
    def get_new_features(self, train_set: DataFrame, test_set: DataFrame) -> DataFrame:
        
        """
        method: get_new_features 
        objective:
                The following code creates features that would be helpful to describe the profile of the customer 
            recodes the customer's education level to numeric form (0: high-school, 1: diploma, 2: bachelors, 3: masters, and 4: doctorates)
            creates a new field to store the household size """
        
        train_set_with_new_features = pd.DataFrame()
        test_set_with_new_features = pd.DataFrame()
        
        datasets = {"train_set": train_set, "test_set": test_set}
    
        for key in datasets:
            dataset = datasets[key]
            
            ##  creating a new field to store the Age of the customer
            dataset['Age']=2022-dataset['Year_Birth']   

            ###  recoding the customer's education level to numeric form (0: high-school, 1: diploma, 2: bachelors, 3: masters, and 4: doctorates)
            dataset["Education"].replace({"Basic":0,"2n Cycle":1, "Graduation":2, "Master":3, "PhD":4},inplace=True)  

            #  recoding the customer's marital status to numeric form (0: not living with a partner, 1: living with a partner) 
            dataset['Marital_Status'].replace({"Married":1, "Together":1, "Absurd":0, "Widow":0, "YOLO":0, "Divorced":0, "Single":0,"Alone":0},inplace=True) 

            #  creating a new field to store the number of children in the household
            dataset['Children']=dataset['Kidhome']+dataset['Teenhome']

            #creating Family_Size
            dataset['Family_Size']=dataset['Marital_Status']+dataset['Children']+1



            #  creating a new field to store the total spending of the customer
            dataset['Total_Spending']=dataset["MntWines"]+ dataset["MntFruits"]+ dataset["MntMeatProducts"]+ dataset["MntFishProducts"]+ dataset["MntSweetProducts"]+ dataset["MntGoldProds"]
            dataset["Total Promo"] =  dataset["AcceptedCmp1"]+ dataset["AcceptedCmp2"]+ dataset["AcceptedCmp3"]+ dataset["AcceptedCmp4"]+ dataset["AcceptedCmp5"]

            ## The following code works out how long the customer has been with the company and store the total number of promotions the customers responded to
            dataset['Dt_Customer']=pd.to_datetime(dataset['Dt_Customer'])
            today=datetime.today()
            dataset['Days_as_Customer']=(today-dataset['Dt_Customer']).dt.days
            dataset['Offers_Responded_To']=dataset['AcceptedCmp1']+dataset['AcceptedCmp2']+dataset['AcceptedCmp3']+dataset['AcceptedCmp4']+dataset['AcceptedCmp5']+dataset['Response']
            dataset["Parental Status"] = np.where(dataset["Children"] > 0, 1, 0)

            #dropping columns which are already used to create new features
            columns_to_drop = ['Year_Birth','Kidhome','Teenhome']
            dataset.drop(columns = columns_to_drop, axis = 1, inplace=True)
            dataset.rename(columns={"Marital_Status": "Marital Status","MntWines": "Wines","MntFruits":"Fruits",
                            "MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets",
                            "MntGoldProds":"Gold","NumWebPurchases": "Web","NumCatalogPurchases":"Catalog",
                            "NumStorePurchases":"Store","NumDealsPurchases":"Discount Purchases"},
                    inplace = True)

            dataset = dataset[
                ["Age","Education","Marital Status","Parental Status",
                "Children","Income","Total_Spending","Days_as_Customer",
                "Recency","Wines","Fruits","Meat","Fish","Sweets","Gold",
                "Web","Catalog","Store","Discount Purchases","Total Promo",
                "NumWebVisitsMonth"]]
            
            if key == "train_set":
                train_set_with_new_features = dataset
            else:
                test_set_with_new_features = dataset        
        logging.info("New features has been created.")
        return train_set_with_new_features, test_set_with_new_features
                
    


