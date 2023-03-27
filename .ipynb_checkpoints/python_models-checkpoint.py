import pandas as pd
import numpy as np
from keras import Sequential, Input, Flatten, Dense


import prepare_data_covid as datacovid

(X_train, y_train), (X_test, y_test), (X_val, y_val) = \
datacovid.prepare_dynamic(df_input = df, window = 3)
(X_train, y_train), (X_test, y_test), (X_val, y_val) =\
    (X_train, y_train), (X_test[1:], y_test[1:]), (X_val[1:], y_val[1:])
X_train.shape

print("Hello")

BATCH_SIZE = 4
PARAMS_LENGTH=X_train.shape[2]
TARGET_LENGTH=y_train.shape[1]
WINDOW=3

model = Sequential(name="Flatten5Layers_Mixed_batch4")
model.add(Input((WINDOW, PARAMS_LENGTH), batch_size=BATCH_SIZE))
model.add(Flatten())
model.add(Dense(WINDOW*PARAMS_LENGTH, activation="relu"))
model.add(Dense(32, activation="linear"))
model.add(Dense(32, activation="linear"))
model.add(Dense(32, activation="linear"))
model.add(Dense(TARGET_LENGTH, activation="linear"))
model.compile(optimizer="adam", loss="mse")
model.summary()
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=BATCH_SIZE)
show_results(X_test, y_test, model, history, batch_size=BATCH_SIZE)