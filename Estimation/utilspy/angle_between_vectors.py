import numpy as np

def angle_between_vectors(v, w):
    # dot_product = np.dot(v, w)  # Calculate the dot product
    # magnitude_v = np.linalg.norm(v)  # Calculate the magnitude of vector v
    # magnitude_w = np.linalg.norm(w)  # Calculate the magnitude of vector w
    # cosine_angle = dot_product / (magnitude_v * magnitude_w)  # Calculate the cosine of the angle
    # angle = np.arccos(cosine_angle)  # Calculate the angle in radians

    dot_product = np.dot(v, w)  # Calculate the dot product
    cross_product = np.linalg.norm(np.cross(v, w))  # Calculate the magnitude of the cross product
    angle = np.arctan2(cross_product, dot_product)  # Calculate the angle using arctan2
    
    return angle