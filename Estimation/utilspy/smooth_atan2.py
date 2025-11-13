import numpy as np

def smooth_atan2(y, x):
    angle = np.arctan2(y, x)
    idx1 = angle < -np.pi
    idx2 = angle > np.pi
    angle[idx1] = angle[idx1] + 2 * np.pi
    angle[idx2] = angle[idx2] - 2 * np.pi
    return angle