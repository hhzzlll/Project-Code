import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal, norm
import seaborn as sns

n_samples = 100000
mean0 = [0, 0]
mean1 = [0, -1]
co_var0 = np.eye(2) * 0.1
co_var1 = np.eye(2) * 0.1

S0 = multivariate_normal.rvs(mean=mean0, cov=co_var0, size=n_samples)
S1 = multivariate_normal.rvs(mean=mean1, cov=co_var1, size=n_samples)
D = S1 - S0

Z1 = np.arctan2(D[:, 0], D[:, 1])
# Fit normal distribution
mu1, sigma1 = norm.fit(Z1)
pd1 = norm(mu1, sigma1)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.hist(Z1, bins=50, density=True, alpha=0.7, label='Data')
x_range = np.linspace(Z1.min(), Z1.max(), 100)
plt.plot(x_range, pd1.pdf(x_range), 'r-', label=f'Normal fit (μ={mu1:.3f}, σ={sigma1:.3f})')
plt.legend()
plt.title('Histogram with Normal Fit for Z1')

plt.subplot(2, 1, 2)
stats.probplot(Z1, dist=pd1, plot=plt)
plt.title('Q-Q Plot for Z1')
plt.tight_layout()
plt.show()

# D and P are equivalent
mean_diff = np.array(mean1) - np.array(mean0)
cov_sum = co_var1 + co_var0
P = multivariate_normal.rvs(mean=mean_diff, cov=cov_sum, size=n_samples)
Z2 = np.arctan2(P[:, 0], P[:, 1])

mu2, sigma2 = norm.fit(Z2)
pd2 = norm(mu2, sigma2)
print(f"pd2.sigma: {sigma2}")

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.hist(Z2, bins=50, density=True, alpha=0.7, label='Data')
x_range = np.linspace(Z2.min(), Z2.max(), 100)
plt.plot(x_range, pd2.pdf(x_range), 'r-', label=f'Normal fit (μ={mu2:.3f}, σ={sigma2:.3f})')
plt.legend()
plt.title('Histogram with Normal Fit for Z2')

plt.subplot(2, 1, 2)
stats.probplot(Z2, dist=pd2, plot=plt)
plt.title('Q-Q Plot for Z2')
plt.tight_layout()
plt.show()
