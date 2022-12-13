import os

import numpy as np
import pandas as pd
import pickle 

from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Plots
import matplotlib.pyplot as plt

# Prerocessing for FEDOT
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

# FEDOT 
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

import logging
logging.raiseExceptions = False

df = None
with open("covid_flow.pkl", "rb") as f:
    df = pickle.load(f)
df["case"] = df.index
df["t_point"] = df["t_point"].apply(lambda x: x[2:])
df["t_point"] = df["t_point"].apply(lambda x: int(x))
df = df.set_index(["case", "t_point"])
df = df.sort_values(["case", "t_point"])

def prepare_data_with_dynamic_window(df_input, window=1, test_size=0.2):
    X, y = [], []
    df = df_input.copy()
    df_grouped = df.groupby(["case"]).size()
    df_idx = df_grouped[df_grouped>=window+1].index
    for i in df_idx:
        for j in range(len(df.loc[(i)])-window):
            wind = df.loc[(i, j): (i, j+window-1)]
            X.append(wind.iloc[:, :47])
            y.append(df.loc[(i, j+window)].iloc[29:42])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return (X_train, y_train), (X_test, y_test)

for WINDOW in range(3, 4):
    (X_train, y_train), (X_test, y_test) = prepare_data_with_dynamic_window(df, window=WINDOW)
    print("Test train shapes:", X_train.shape,
    X_test.shape)
    X_train = X_train.reshape(-1, WINDOW*47)
    X_test = X_test.reshape(-1, WINDOW*47)

    models = {}
    dinam_fact_columns = []
    dinam_fact_columns.append(df.columns[29:42][4])
    dinam_fact_columns.append(df.columns[29:42][2])

    for param_idx, param_name in enumerate(dinam_fact_columns):
        model = Fedot(problem='regression', timeout=10, n_jobs=-1)
        try:
            obtained_pipeline = model.fit(features=X_train, target=y_train[:, param_idx])
            models[param_name] = (model, obtained_pipeline)
            obtained_pipeline.save(f"fedot_w{WINDOW}/fedot_regress_param_{param_idx}.json", datetime_in_path=False)  
            print(f"SAVED {param_name}")
        except:
            try:
                obtained_pipeline = model.fit(features=X_train, target=y_train[:, param_idx])
                models[param_name] = (model, obtained_pipeline)
                obtained_pipeline.save(f"fedot_w{WINDOW}/fedot_regress_param_{param_idx}.json", datetime_in_path=False)  
                print(f"SAVED 2 attempt {param_name}")
            except Exception as e:
                print(f"FAILED {param_name} {e}")
    

