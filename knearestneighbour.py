import numpy as np
from scipy import stats

class knnclassifier:

    def __init__(self, X, y, k):
        self.X = np.array(X)
        self.y = np.array(y)
        self.m = np.array(X).shape[0]
        self.k = k


    def predict(self):

        y_pred = np.zeros(self.m)
        for i in range(self.m):
            temp = np.delete(self.X, i, axis = 0)
            distance = np.sum(np.abs(temp - self.X[i, :]), axis = 1)
            distance_sorted = np.sort(distance)
            nearest_k = distance_sorted[:k]
            nearest_k_y = np.argsort(nearest_k)
            nearest_k_values = [self.y[j] for j in nearest_k_y]
            y_pred[i] = np.round(np.mean(nearest_k_values))

        acc = np.mean(y_pred == self.y)
        cache = {"output" : y_pred, "accuracy" : acc}
