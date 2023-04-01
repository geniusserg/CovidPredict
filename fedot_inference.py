#####
# Patient state prediction model
# Algorithm uses pre-fitted Fedot models
# All models are in folder "fedot pipelines"
#
# Input - csv file "input.csv" with parameters of patinets
#
# Output - csv file "output.csv" with predictions
#
#####

import os

import numpy as np
import pandas as pd


from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

class FedotCovidPredict():
    def __init__(self, models_directory:str = "fedot_pipelines"):
        self.models = os.listdir("fedot_pipelines")
        self.pipelines = {}
        for model in self.models:
            pipeline = Pipeline()
            pipeline.load(f"fedot_pipelines/{model}/{model}.json")
            self.pipelines[model] = pipeline
    
    def predict(self, df:pd.DataFrame):
        state_vector = df.groupby("case").last() # use only last state vector
        input_data = InputData(idx=state_vector.index, 
                  features=state_vector, 
                  data_type=DataTypesEnum.table,
                  task=Task(TaskTypesEnum.regression))

        prediction_vector = {}
        for model in models:
            prediction_vector[model] = pipelines[model].predict(input_data).predict #sequentially run pipelines

        # Round categorical features 
        result = pd.DataFrame(prediction_vector)
        result.loc[:, "снижение_сознания_dinam_fact"] = result["снижение_сознания_dinam_fact"].apply(lambda x: int(x))
        result.loc[:, "Cтепень тяжести по КТ_dinam_fact"] = result["Cтепень тяжести по КТ_dinam_fact"].apply(lambda x: int(x))
        return result 
        
        
