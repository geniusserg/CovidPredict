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

test = [[36.6,1.9,41,72,19,10,9,0.024,18,0.0098,3,347,14],
[36.6,1.9,41,72,19,10,9,0.024,18,0.0098,3,347,14],
[36.6,1.9,41,72,19,10,9,0.024,18,0.0098,3,347,14]]

model = CovidPatientStatePredictor(model_path="model1")
result = model.predict(test)

print(f"Predictet patinet state: {result}")