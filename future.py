class Future:

    def __init__(self, index, observations, node_times, predicted_nodes, pred_goal):
        self.index = index
        self.observations = observations
        self.node_times = node_times
        self.nodes = predicted_nodes
        self.goal = pred_goal

