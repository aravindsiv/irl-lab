import numpy as np

def feature_averages(trajectory,gamma=0.99):
    horizon = len(trajectory)
    return np.sum(np.multiply(trajectory,np.array([gamma**j for j in range(horizon)]).reshape(horizon,1)),axis=0)