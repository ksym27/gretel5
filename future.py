import numpy


class Future:

    def __init__(self, size):
        self.size = size

        self.observations = [None] * size
        self.observation_times = [None] * size
        self.observation_steps = [None] * size

        self.predicted_nodes = [None] * size
        self.predicted_times = [None] * size

        self.condition = numpy.zeros(size, dtype=numpy.int)
