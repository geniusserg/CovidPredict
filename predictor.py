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
        scaler_file = open("models/standard_scaler.pkl", 'rb')
        self.imputer = pickle.load(imputer_file)
        self.scaler  = pickle.load(scaler_file)
        self.columns = []
        self.index   = []

    def _prepare_time_series(self, df: pandas.DataFrame):
        dinam_fact_df = df.iloc[:,29:42]
        stat_fact_df  = df.iloc[:,:29]
        stat_fact_df  = pd.concat([stat_fact_df, df.iloc[:, 42]])
        stat_fact_df  = pd.concat([stat_fact_df, df.iloc[:, 42:]])
        df_imputed = self.imputer.fit_transform(dinam_fact_df)
        df_scaled = self.scaler.transform(df_imputed)
        dinam_fact_df = pd.DataFrame(data=df_scaled, columns = dinam_fact_df.columns, index=dinam_fact_df.index)
        return dinam_fact_df, stat_fact_df

    def _inverse_transform_data(self, Y_scaled_predicted):
        y_pred_inv = self.scaler.inverse_transform(Y_scaled_predicted)[0]
        result_df = pd.DataFrame(data=y_pred_inv, columns=self.columns[29:42])
        return  result_df

    def predict(self, df):
        self.columns = df.columns
        self.index = df.index
        dinam_df, stat_df = self._prepare_time_series(df)
        predicted_data = self.model.predict(np.array(dinam_df))
        inveresed_predicted_data = self._inverse_transform_data(predicted_data)
        return inveresed_predicted_data

if __name__=="__main__":
    model = CovidPatientStatePredictor("models/LSTM13")
    df = pd.read_csv("example/input.csv")
    result = model.predict(df)
    result.to_csv("example/output.csv")
