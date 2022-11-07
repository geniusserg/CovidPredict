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
        imputer_file = open(os.path.sep.join([model_path, 'imputer.pkl']), 'rb')
        scaler_file = open(os.path.sep.join([model_path, 'scaler.pkl']), 'rb')
        self.imputer = pickle.load(imputer_file)
        self.scaler  = pickle.load(scaler_file)

    def _impute_nan(self, df):
        return self.imputer.fit_transform(X)

    def _scale_transform_data(self, X):
        return self.scaler.transform(X)

    def _inverse_transform_data(self, Y_scaled_predicted):
        y_pred_inv = self.scaler.inverse_transform(Y_scaled_predicted)[0]
        y_unnorm_pred = [float("{:0.2f}".format(i)) for i in y_pred_inv]
        return y_unnorm_pred

    def _prepare_time_series(self, df: pandas.DataFrame):
        return df

    def set_model(self, model_path):
        self.model = load_model(model_path)

    def predict(self, df, to_csv=True):
        prepared_data = self._prepare_time_series(df)
        imputed_data = self._impute_nan(prepared_data)
        scaled_transformed_data = self._scale_transform_data(imputed_data)
        predicted_data = self.model.predict(np.array([scaled_transformed_data]))
        inveresed_predicted_data = self._inverse_transform_data(predicted_data)
        return inveresed_predicted_data

if __name__=="__main__":
    model = CovidPatientStatePredictor("LSTM13")
    df = pd.read_csv("example/input.csv")
    result = model.predict(df)
    result.to_csv("example/output.csv")
