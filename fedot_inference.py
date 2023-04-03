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
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.pipeline import Pipeline

class FedotCovidPredict():
    def __init__(self, models_directory:str = "fedot_pipelines_april"):
        self.models = os.listdir(models_directory)
        self.pipelines = {} # here stored fitted Pipelines
        for model in self.models:
            pipeline = Pipeline()
            pipeline.load(f"{models_directory}/{model}/{model}.json")
            self.pipelines[model] = pipeline
    
    def predict(self, df:pd.DataFrame):
        state_vector = df.groupby("case").last() #  Our models use only one vector
        input_data = InputData(idx=state_vector.index,  # transform to InputData
                  features=state_vector,
                  data_type=DataTypesEnum.table,
                  task=Task(TaskTypesEnum.regression))

        #sequentially run pipelines on all parameters
        prediction_vector = {}
        for model in self.models:
            prediction_vector[model] = self.pipelines[model].predict(input_data).predict

        # Round values of categorical features
        result = pd.DataFrame(prediction_vector)
        result.loc[:, "снижение_сознания_dinam_fact"] = result["снижение_сознания_dinam_fact"].apply(lambda x: int(x))
        result.loc[:, "Cтепень тяжести по КТ_dinam_fact"] = result["Cтепень тяжести по КТ_dinam_fact"].apply(lambda x: int(x))
        return result