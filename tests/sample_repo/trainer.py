from utils.helper import clean_data

class Trainer:
    def __init__(self):
        self.model = None

    def train(self, data):
        data = clean_data(data)
        print("Training...")
