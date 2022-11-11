import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from keras.models import load_model
import numpy as np
import pandas 
import pickle 
import os 

class CovidPatientStatePredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        imputer_file = open("models/iterative_imputer.pkl", 'rb')
        scaler_file = open("models/minmax_scaler.pkl", 'rb')
        self.imputer = pickle.load(imputer_file)
        self.scaler  = pickle.load(scaler_file)
        self.columns = self._set_columns()
        self.index   = []
    
    def _set_columns(self):
        columns = []
        with open("columns_target.txt", "r") as f:
            columns = f.read().split("\n")
        return columns
        
    def _prepare_time_series(self, df: pandas.DataFrame):
        df.set_index(["case", "t_point"], inplace=True)
        df.sort_values(["case", "t_point"], inplace=True)
        dinam_fact_df = df.iloc[:,29:42]
        stat_fact_df  = df.iloc[:,:29]
        stat_fact_df  = pd.concat([stat_fact_df, df.iloc[:, 42]])
        stat_fact_df  = pd.concat([stat_fact_df, df.iloc[:, 42:]])
        self.index = df.index.levels[0]
        df_scaled = self.scaler.transform(dinam_fact_df)
        df_imputed = self.imputer.transform(df_scaled)
        df_imputed = df_imputed.reshape(-1, 3, 13)
        return df_imputed, stat_fact_df

    def _inverse_transform_data(self, Y_scaled_predicted):
        y_pred_inv = self.scaler.inverse_transform(Y_scaled_predicted)
        result_df = pd.DataFrame(data=y_pred_inv, columns=self.columns, index=self.index)
        return result_df

    def predict(self, df):
        dinam_df, stat_df = self._prepare_time_series(df)
        predicted_data = self.model.predict(dinam_df, batch_size=1)
        inveresed_predicted_data = self._inverse_transform_data(predicted_data)
        return inveresed_predicted_data

if __name__=="__main__":
    model = CovidPatientStatePredictor("models/minmax_scaler/RNN13")
    df = pd.read_csv("example/input.csv")
    result = model.predict(df)
    result.to_csv("example/output.csv")
