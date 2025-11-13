import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import serial
import time
import json

class SensorVerifier:
    def __init__(self, config=None):
        """Initialize sensor verifier"""
        self.config = config or {
            'sampling_rate': 100,  # Hz
            'test_duration': 10,   # seconds
            'noise_threshold': 0.1,
            'drift_threshold': 0.05
        }
        self.sensor_data = {}
        
    def connect_sensor(self, port, baudrate=115200):
        """Connect to sensor via serial port"""
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for connection
            print(f"Connected to sensor on {port}")
            return True
        except Exception as e:
            print(f"Failed to connect to sensor: {e}")
            return False
    
    def read_sensor_data(self, duration=None):
        """Read sensor data for specified duration"""
        if duration is None:
            duration = self.config['test_duration']
            
        print(f"Reading sensor data for {duration} seconds...")
        
        data = {
            'timestamp': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
            'mag_x': [], 'mag_y': [], 'mag_z': []
        }
        
        start_time = time.time()
        while time.time() - start_time < duration:
            if hasattr(self, 'serial_conn') and self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode('utf-8').strip()
                parsed_data = self.parse_sensor_line(line)
                
                if parsed_data:
                    data['timestamp'].append(time.time() - start_time)
                    for key, value in parsed_data.items():
                        if key in data:
                            data[key].append(value)
            else:
                # Generate synthetic data for testing
                t = time.time() - start_time
                data['timestamp'].append(t)
                data['accel_x'].append(np.sin(2*np.pi*t) + np.random.normal(0, 0.1))
                data['accel_y'].append(np.cos(2*np.pi*t) + np.random.normal(0, 0.1))
                data['accel_z'].append(9.81 + np.random.normal(0, 0.1))
                data['gyro_x'].append(np.random.normal(0, 0.05))
                data['gyro_y'].append(np.random.normal(0, 0.05))
                data['gyro_z'].append(np.random.normal(0, 0.05))
                data['mag_x'].append(0.2 + np.random.normal(0, 0.02))
                data['mag_y'].append(0.3 + np.random.normal(0, 0.02))
                data['mag_z'].append(0.4 + np.random.normal(0, 0.02))
                
            time.sleep(1/self.config['sampling_rate'])
        
        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
            
        self.sensor_data = data
        return data
    
    def parse_sensor_line(self, line):
        """Parse sensor data line (customize based on sensor format)"""
        try:
            # Example format: "accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z"
            values = [float(x) for x in line.split(',')]
            if len(values) >= 9:
                return {
                    'accel_x': values[0], 'accel_y': values[1], 'accel_z': values[2],
                    'gyro_x': values[3], 'gyro_y': values[4], 'gyro_z': values[5],
                    'mag_x': values[6], 'mag_y': values[7], 'mag_z': values[8]
                }
        except:
            pass
        return None
    
    def verify_sampling_rate(self):
        """Verify actual sampling rate"""
        if 'timestamp' not in self.sensor_data:
            print("No timestamp data available")
            return False
            
        timestamps = self.sensor_data['timestamp']
        if len(timestamps) < 2:
            print("Insufficient data for sampling rate verification")
            return False
            
        # Calculate actual sampling rate
        dt = np.diff(timestamps)
        actual_rate = 1.0 / np.mean(dt)
        expected_rate = self.config['sampling_rate']
        
        rate_error = abs(actual_rate - expected_rate) / expected_rate
        
        print(f"Expected sampling rate: {expected_rate:.1f} Hz")
        print(f"Actual sampling rate: {actual_rate:.1f} Hz")
        print(f"Rate error: {rate_error*100:.1f}%")
        
        if rate_error < 0.05:  # 5% tolerance
            print("✓ Sampling rate verification passed")
            return True
        else:
            print("⚠ Sampling rate verification failed")
            return False
    
    def verify_noise_levels(self):
        """Verify sensor noise levels"""
        print("Verifying noise levels...")
        
        results = {}
        for sensor_type in ['accel', 'gyro', 'mag']:
            axes = ['x', 'y', 'z']
            sensor_results = {}
            
            for axis in axes:
                key = f"{sensor_type}_{axis}"
                if key in self.sensor_data:
                    data = self.sensor_data[key]
                    
                    # Calculate noise metrics
                    noise_std = np.std(data)
                    noise_rms = np.sqrt(np.mean(data**2))
                    
                    sensor_results[axis] = {
                        'std': noise_std,
                        'rms': noise_rms,
                        'passed': noise_std < self.config['noise_threshold']
                    }
                    
            results[sensor_type] = sensor_results
            
        # Print results
        for sensor_type, sensor_results in results.items():
            print(f"\n{sensor_type.upper()} noise analysis:")
            for axis, metrics in sensor_results.items():
                status = "✓" if metrics['passed'] else "⚠"
                print(f"  {axis}: std={metrics['std']:.4f}, rms={metrics['rms']:.4f} {status}")
                
        return results
    
    def verify_drift(self):
        """Verify sensor drift"""
        print("Verifying sensor drift...")
        
        results = {}
        for sensor_type in ['accel', 'gyro', 'mag']:
            axes = ['x', 'y', 'z']
            sensor_results = {}
            
            for axis in axes:
                key = f"{sensor_type}_{axis}"
                if key in self.sensor_data:
                    data = self.sensor_data[key]
                    
                    # Calculate linear trend (drift)
                    time_points = np.arange(len(data))
                    slope, intercept = np.polyfit(time_points, data, 1)
                    
                    # Drift in units per second
                    drift_rate = slope * self.config['sampling_rate']
                    
                    sensor_results[axis] = {
                        'drift_rate': drift_rate,
                        'passed': abs(drift_rate) < self.config['drift_threshold']
                    }
                    
            results[sensor_type] = sensor_results
            
        # Print results
        for sensor_type, sensor_results in results.items():
            print(f"\n{sensor_type.upper()} drift analysis:")
            for axis, metrics in sensor_results.items():
                status = "✓" if metrics['passed'] else "⚠"
                print(f"  {axis}: drift={metrics['drift_rate']:.6f} units/s {status}")
                
        return results
    
    def plot_sensor_data(self):
        """Plot sensor data for visual inspection"""
        if not self.sensor_data:
            print("No sensor data to plot")
            return
            
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Sensor Data Verification')
        
        timestamps = self.sensor_data['timestamp']
        
        # Accelerometer
        for i, axis in enumerate(['x', 'y', 'z']):
            key = f"accel_{axis}"
            if key in self.sensor_data:
                axes[0, i].plot(timestamps, self.sensor_data[key])
                axes[0, i].set_title(f'Accelerometer {axis.upper()}')
                axes[0, i].set_ylabel('Acceleration (m/s²)')
                axes[0, i].grid(True, alpha=0.3)
        
        # Gyroscope
        for i, axis in enumerate(['x', 'y', 'z']):
            key = f"gyro_{axis}"
            if key in self.sensor_data:
                axes[1, i].plot(timestamps, self.sensor_data[key])
                axes[1, i].set_title(f'Gyroscope {axis.upper()}')
                axes[1, i].set_ylabel('Angular velocity (rad/s)')
                axes[1, i].grid(True, alpha=0.3)
        
        # Magnetometer
        for i, axis in enumerate(['x', 'y', 'z']):
            key = f"mag_{axis}"
            if key in self.sensor_data:
                axes[2, i].plot(timestamps, self.sensor_data[key])
                axes[2, i].set_title(f'Magnetometer {axis.upper()}')
                axes[2, i].set_ylabel('Magnetic field (T)')
                axes[2, i].set_xlabel('Time (s)')
                axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_verification_suite(self, port=None):
        """Run complete sensor verification suite"""
        print("Starting sensor verification suite...")
        
        # Connect to sensor if port provided
        if port:
            if not self.connect_sensor(port):
                print("Failed to connect to sensor, using synthetic data")
        
        # Read sensor data
        self.read_sensor_data()
        
        # Run verification tests
        print("\n" + "="*50)
        rate_ok = self.verify_sampling_rate()
        
        print("\n" + "="*50)
        noise_results = self.verify_noise_levels()
        
        print("\n" + "="*50)
        drift_results = self.verify_drift()
        
        # Plot results
        self.plot_sensor_data()
        
        # Generate report
        self.generate_verification_report(rate_ok, noise_results, drift_results)
        
        return rate_ok, noise_results, drift_results
    
    def generate_verification_report(self, rate_ok, noise_results, drift_results):
        """Generate verification report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sampling_rate_ok': rate_ok,
            'noise_results': noise_results,
            'drift_results': drift_results,
            'overall_pass': rate_ok and self.all_tests_passed(noise_results, drift_results)
        }
        
        # Save report
        with open('sensor_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\nVerification report saved to sensor_verification_report.json")
        print(f"Overall result: {'✓ PASS' if report['overall_pass'] else '⚠ FAIL'}")
        
    def all_tests_passed(self, noise_results, drift_results):
        """Check if all tests passed"""
        for sensor_results in noise_results.values():
            for axis_results in sensor_results.values():
                if not axis_results['passed']:
                    return False
                    
        for sensor_results in drift_results.values():
            for axis_results in sensor_results.values():
                if not axis_results['passed']:
                    return False
                    
        return True

# Example usage
if __name__ == "__main__":
    verifier = SensorVerifier()
    
    # Run verification (replace with actual serial port if available)
    verifier.run_verification_suite()  # Use synthetic data
    # verifier.run_verification_suite("COM3")  # Use real sensor
