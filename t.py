import argparse
import os
import keras
from keras.models import load_model
import pandas as pd
import numpy as np 

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help = "Model")
args = vars(ap.parse_args())
model = load_model(args["model"])
print(model.input)

test_df = np.array([[36.6,1.9,41,72,19,10,9,0.024,18,0.0098,3,347,14],
[36.6,1.9,41,72,19,10,9,0.024,18,0.0098,3,347,14],
[36.6,1.9,41,72,19,10,9,0.024,18,0.0098,3,347,14]])

print(test_df.shape)
print(model.predict(test_df))