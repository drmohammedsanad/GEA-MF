import numpy as np


class PopularityModel:
    """
    Simple popularity-based recommender

    Very important baseline:
    ranks items by overall popularity
    """

    def __init__(self):
        self.item_scores = None

    def train(self, R):
        # Count how many interactions each item has
        self.item_scores = np.sum(R, axis=0)

    def predict_scores(self):
        return self.item_scores
