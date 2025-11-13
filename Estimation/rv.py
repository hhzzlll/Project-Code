import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal, norm, chi2
import matplotlib.pyplot as plt

class RandomVariable:
    """Random variable utilities for pose estimation"""
    
    @staticmethod
    def sample_multivariate_normal(mean, cov, size=1):
        """Sample from multivariate normal distribution"""
        return multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    
    @staticmethod
    def pdf_multivariate_normal(x, mean, cov):
        """Probability density function of multivariate normal"""
        return multivariate_normal.pdf(x, mean=mean, cov=cov)
    
    @staticmethod
    def mahalanobis_distance(x, mean, cov):
        """Calculate Mahalanobis distance"""
        diff = x - mean
        inv_cov = np.linalg.pinv(cov)
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    
    @staticmethod
    def confidence_ellipse(mean, cov, confidence=0.95):
        """Generate confidence ellipse parameters"""
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
        
        # Chi-square value for confidence level
        chi2_val = chi2.ppf(confidence, df=2)
        
        # Ellipse parameters
        a = np.sqrt(chi2_val * eigenvals[0])
        b = np.sqrt(chi2_val * eigenvals[1])
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        
        return a, b, angle
    
    @staticmethod
    def plot_confidence_ellipse(mean, cov, confidence=0.95, ax=None, **kwargs):
        """Plot confidence ellipse"""
        if ax is None:
            fig, ax = plt.subplots()
            
        a, b, angle = RandomVariable.confidence_ellipse(mean, cov, confidence)
        
        # Generate ellipse points
        theta = np.linspace(0, 2*np.pi, 100)
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)
        
        # Rotate ellipse
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        x_rot = cos_angle * x_ellipse - sin_angle * y_ellipse
        y_rot = sin_angle * x_ellipse + cos_angle * y_ellipse
        
        # Translate to mean
        x_rot += mean[0]
        y_rot += mean[1]
        
        ax.plot(x_rot, y_rot, **kwargs)
        return ax

# Example usage
if __name__ == "__main__":
    # Test random variable utilities
    rv = RandomVariable()
    
    # Generate samples
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 2]]
    samples = rv.sample_multivariate_normal(mean, cov, 1000)
    
    # Plot samples and confidence ellipse
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, label='Samples')
    rv.plot_confidence_ellipse(mean, cov, 0.95, color='red', linewidth=2, label='95% Confidence')
    rv.plot_confidence_ellipse(mean, cov, 0.68, color='orange', linewidth=2, label='68% Confidence')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Multivariate Normal Distribution with Confidence Ellipses')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
