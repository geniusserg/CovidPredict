import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

##################################################################
#
# Prepare data with fixed window from start with only dinamic data
#
# Window starts from Start
# MinMaxScaler
# IterativeImputer
#
##################################################################
def prepare_dynamic(df_input, window=3, test_size=0.2):
    df = df_input.copy()
    X, y = [], []
    df_grouped = df.groupby(["case"]).size()
    df_idx = df_grouped[df_grouped>=window+1].index
    scal_model = MinMaxScaler()
    df_scaled = scal_model.fit_transform(df.iloc[:, 29:43])
    imputer = IterativeImputer()
    df_imputed = imputer.fit_transform(df_scaled)
    df.iloc[:, 29:43] = df_imputed
    for i in df_idx:
        for j in range(len(df.loc[(i)])-window):
            wind = df.loc[(i, j): (i, j+window-1)]
            X.append(wind.iloc[:, 29:42])
            y.append(df.loc[(i, j+window)].iloc[29:42])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=1/2)
    print("TRAIN shape: ", X_train.shape)
    print("TEST shape: ", y_test.shape)
    print("VAL shape: ", y_val.shape)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

##################################################################
#
# Prepare data with fixed window with static and dynamic data
#
# Window starts from any point
# MinMaxScaler
# IterativeImputer
#
##################################################################
def prepare_data_with_static(df_input, window=3, test_size=0.2):
    X, y = [], []
    df = df_input.copy()
    df_grouped = df.groupby(["case"]).size()
    df_idx = df_grouped[df_grouped>=window+1].index
    scal_model = MinMaxScaler()
    df_scaled = scal_model.fit_transform(df.iloc[:, 29:43])
    imputer = IterativeImputer(max_iter=50)
    df_imputed = imputer.fit_transform(df_scaled)
    df.iloc[:, 29:43] = df_imputed
    for i in df_idx:
        for j in range(len(df.loc[(i)])-window):
            wind = df.loc[(i, j): (i, j+window-1)]
            X.append(wind.iloc[:, :47])
            y.append(df.loc[(i, window)].iloc[29:42])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=1/2)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def divide_dynamic_static(X_train, X_test, X_val):
    D_train = X_train[:, :, 29:]
    S_train = X_train[:, 0, 0:29]
    S_train = np.hstack([S_train, X_train[:, 0, 42:43]])
    D_val = X_val[:, :, 29:]
    S_val = X_val[:, 0, :29]
    S_val = np.hstack([S_val, X_val[:, 0, 42:43]])
    D_test = X_test[:, :, 29:]
    S_test = X_test[:, 0, :29]
    S_test = np.hstack([S_test, X_test[:, 0, 42:43]])
    return D_train, S_train, D_val, S_val, D_test, S_test

