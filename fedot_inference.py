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
import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.pipeline import Pipeline


class FedotCovidPredict:
    def __init__(self, models_directory: str = "fedot_pipelines", window_length=1):
        self.models = os.listdir(models_directory)
        self.pipelines = {}  # here stored fitted Pipelines
        self.window_length = window_length
        for model in self.models:
            pipeline = Pipeline()
            pipeline.load(f"{models_directory}/{model}/{model}.json")
            self.pipelines[model] = pipeline

    def predict(self, df: pd.DataFrame):
        state_vector = df.groupby("case").tail(self.window_length)  # Our models use only one vector
        state_vector = np.array(state_vector).reshape(-1, self.window_length * 47)
        input_data = InputData(idx=state_vector.index,  # transform to InputData
                               features=state_vector,
                               data_type=DataTypesEnum.table,
                               task=Task(TaskTypesEnum.regression))

        # sequentially run pipelines on all parameters
        prediction_vector = {}
        for model in self.models:
            prediction_vector[model] = self.pipelines[model].predict(input_data).predict

        # Round values of categorical features
        result = pd.DataFrame(prediction_vector)
        return result
