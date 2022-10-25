

class CovidPatientStatePredictor():
    def __init__(self, model):
        self.model = model
    
    def _preprocess(self, data):
        pass

    def set_model(self, model):
        self.model = model

    def predict(self, data):
        processed_data = self.preprocessor(data)
        predicted_data = self.model.predict(data)

