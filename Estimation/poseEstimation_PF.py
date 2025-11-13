import numpy as np
import matplotlib.pyplot as plt
from myStateTransitionFcn import myStateTransitionFcn
from myMeasurementLikelihoodFcn import myMeasurementLikelihoodFcn

class ParticleFilter:
    def __init__(self, num_particles=1000, initial_state=None, initial_covariance=None):
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, 3))  # [x, y, theta]
        self.weights = np.ones(num_particles) / num_particles
        
        # Initialize particles
        if initial_state is None:
            initial_state = [0, 0, 0]
        if initial_covariance is None:
            initial_covariance = np.eye(3) * 0.1
            
        self.particles = np.random.multivariate_normal(initial_state, initial_covariance, num_particles)
        
    def predict(self, dt=0.1):
        """Prediction step"""
        self.particles = myStateTransitionFcn(self.particles, dt)
        
    def update(self, measurement):
        """Update step"""
        # Calculate likelihood for each particle
        likelihood = myMeasurementLikelihoodFcn(self.particles, measurement)
        
        # Update weights
        self.weights = self.weights * likelihood
        self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        # Resample if necessary
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        if effective_sample_size < self.num_particles / 2:
            self.resample()
            
    def resample(self):
        """Systematic resampling"""
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def get_estimate(self):
        """Get weighted mean estimate"""
        return np.average(self.particles, weights=self.weights, axis=0)
        
    def get_covariance(self):
        """Get weighted covariance"""
        mean = self.get_estimate()
        diff = self.particles - mean
        return np.cov(diff.T, aweights=self.weights)

# Example usage
if __name__ == "__main__":
    # Initialize particle filter
    pf = ParticleFilter(num_particles=1000)
    
    # Simulation parameters
    num_steps = 100
    measurements = []  # Load your measurements here
    
    # Run particle filter
    estimates = []
    for t in range(num_steps):
        # Prediction
        pf.predict()
        
        # Update (if measurement available)
        if t < len(measurements):
            pf.update(measurements[t])
            
        # Store estimate
        estimates.append(pf.get_estimate())
        
    # Plot results
    estimates = np.array(estimates)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(estimates[:, 0])
    plt.title('X Position')
    plt.xlabel('Time Step')
    plt.ylabel('X [m]')
    
    plt.subplot(2, 2, 2)
    plt.plot(estimates[:, 1])
    plt.title('Y Position')
    plt.xlabel('Time Step')
    plt.ylabel('Y [m]')
    
    plt.subplot(2, 2, 3)
    plt.plot(np.rad2deg(estimates[:, 2]))
    plt.title('Orientation')
    plt.xlabel('Time Step')
    plt.ylabel('Theta [deg]')
    
    plt.subplot(2, 2, 4)
    plt.plot(estimates[:, 0], estimates[:, 1])
    plt.title('Trajectory')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
