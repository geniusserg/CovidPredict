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

(X_train, y_train), (X_test, y_test) = prepare_data_with_dynamic_window(df, window=1)
print("Test train shapes:", X_train.shape,
X_test.shape)
X_train = X_train.reshape(-1, 47)
X_test = X_test.reshape(-1, 47)

dinam_fact_columns = df.columns[29:42]
scores = {}
for param_idx, param_name in enumerate(dinam_fact_columns):
    try:
        model = Fedot("regression")
        model.load(f"models\\7December\\fedot_regress_one_window_param_{param_idx}\\fedot_regress_one_window_param_{param_idx}.json")
        y_pred = model.predict(X_test)
        mask = ~np.isnan(y_test[:, param_idx])
        y_check = y_test[mask, param_idx] 
        y_pred = y_pred[mask] 
        scaler = MinMaxScaler()

        scaler.fit(y_check.reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1)).reshape(-1)
        y_check = scaler.transform(y_check.reshape(-1, 1)).reshape(-1)
        r2_res = r2_score(y_check, y_pred, multioutput="raw_values")
        mse = mean_squared_error(y_check, y_pred)
        scores[param_name]=(mse, r2_res[0])
    except Exception as e:
        print(f"FAILED {param_name} {e}")

df = pd.DataFrame(scores, index=["MSE", "R2"], columns = dinam_fact_columns)
df.to_csv("fedot.csv")