import numpy as np

def generate_sample_distribution(mean, p, n_samples):
    sz = len(mean)
    std = 1/(2*np.pi*p)
    co_var = (std**2) * np.eye(sz)
    samples = np.random.multivariate_normal(mean, co_var, n_samples)
    return samples