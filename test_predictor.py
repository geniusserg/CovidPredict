import pickle
from predictor import CovidPatientStatePredictor


df = None
with open("covid_flow.pkl", "rb") as f:
    df = pickle.load(f)
df["case"] = df.index
df["t_point"] = df["t_point"].apply(lambda x: x[2:])
df["t_point"] = df["t_point"].apply(lambda x: int(x))
df.set_index(["case", "t_point"], inplace=True)
df.sort_index()

test = df.loc[("GACAAcY")]

model = CovidPatientStatePredictor(model_path="LSTM64xDense13")
result = model.predict(test)

print(f"Predictet patinet state: {result}")