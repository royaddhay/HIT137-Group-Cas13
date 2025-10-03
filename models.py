from transformers import pipeline

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def run(self, input_data):
        raise NotImplementedError("Subclasses must override run()")

# Polymorphism + Method Overriding
class SentimentModel(BaseModel):
    def __init__(self):
        super().__init__("distilbert-base-uncased-finetuned-sst-2-english")
        self.pipe = pipeline("sentiment-analysis")

    def run(self, input_data):
        return self.pipe(input_data)

class ImageClassificationModel(BaseModel):
    def __init__(self):
        super().__init__("google/vit-base-patch16-224")
        self.pipe = pipeline("image-classification")

    def run(self, input_data):
        return self.pipe(images=input_data)
