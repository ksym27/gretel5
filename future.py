class Future:

    def __init__(self, observations, prediction, start=None, target=None):
        self.observations = observations
        self.start = start
        self.target = target
        self.prediction = prediction

