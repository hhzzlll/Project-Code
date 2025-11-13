import numpy as np
import cv2
import json
import requests
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

class OpenCapTester:
    def __init__(self, config_file=None):
        """Initialize OpenCap tester"""
        self.config = self.load_config(config_file)
        self.camera_params = {}
        self.pose_data = []
        
    def load_config(self, config_file):
        """Load configuration from file"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'camera_count': 2,
                'video_resolution': [1920, 1080],
                'fps': 30,
                'calibration_file': 'calibration.json'
            }
    
    def test_camera_calibration(self, calibration_data):
        """Test camera calibration parameters"""
        print("Testing camera calibration...")
        
        for cam_id, params in calibration_data.items():
            print(f"Camera {cam_id}:")
            print(f"  Intrinsic matrix:\n{params['intrinsic_matrix']}")
            print(f"  Distortion coefficients: {params['distortion_coeffs']}")
            print(f"  Reprojection error: {params.get('reprojection_error', 'N/A')}")
            
            # Validate calibration quality
            if 'reprojection_error' in params:
                if params['reprojection_error'] < 1.0:
                    print(f"  ✓ Good calibration quality")
                else:
                    print(f"  ⚠ Poor calibration quality")
    
    def test_pose_estimation(self, video_files):
        """Test pose estimation on video files"""
        print("Testing pose estimation...")
        
        pose_results = []
        for i, video_file in enumerate(video_files):
            if not os.path.exists(video_file):
                print(f"Video file not found: {video_file}")
                continue
                
            print(f"Processing video {i+1}: {video_file}")
            poses = self.extract_poses_from_video(video_file)
            pose_results.append(poses)
            
        return pose_results
    
    def extract_poses_from_video(self, video_file):
        """Extract poses from video file (placeholder)"""
        # This would integrate with actual pose estimation pipeline
        cap = cv2.VideoCapture(video_file)
        poses = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Placeholder pose detection
            # In real implementation, this would call pose estimation model
            pose = self.dummy_pose_detection(frame, frame_count)
            poses.append(pose)
            
            frame_count += 1
            if frame_count >= 100:  # Limit for testing
                break
                
        cap.release()
        return poses
    
    def dummy_pose_detection(self, frame, frame_num):
        """Dummy pose detection for testing"""
        # Generate synthetic pose data
        num_keypoints = 17  # COCO format
        pose = {
            'frame': frame_num,
            'timestamp': frame_num / 30.0,  # Assuming 30 fps
            'keypoints': np.random.rand(num_keypoints, 3) * 100,  # x, y, confidence
            'bbox': [100, 100, 200, 300]  # x, y, width, height
        }
        return pose
    
    def validate_pose_data(self, poses):
        """Validate pose estimation results"""
        print("Validating pose data...")
        
        if not poses:
            print("⚠ No pose data found")
            return False
            
        # Check data consistency
        frame_count = len(poses)
        print(f"Total frames processed: {frame_count}")
        
        # Check for missing data
        missing_frames = []
        for i, pose in enumerate(poses):
            if pose is None or len(pose['keypoints']) == 0:
                missing_frames.append(i)
                
        if missing_frames:
            print(f"⚠ Missing pose data in {len(missing_frames)} frames")
        else:
            print("✓ All frames have pose data")
            
        return len(missing_frames) < frame_count * 0.1  # Less than 10% missing
    
    def plot_pose_trajectory(self, poses, joint_name='nose'):
        """Plot trajectory of specific joint"""
        if not poses:
            return
            
        # Extract joint positions (assuming nose is keypoint 0)
        joint_idx = 0  # nose keypoint
        x_coords = []
        y_coords = []
        
        for pose in poses:
            if pose and len(pose['keypoints']) > joint_idx:
                x_coords.append(pose['keypoints'][joint_idx, 0])
                y_coords.append(pose['keypoints'][joint_idx, 1])
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(x_coords, label='X coordinate')
        plt.plot(y_coords, label='Y coordinate')
        plt.xlabel('Frame')
        plt.ylabel('Pixel coordinates')
        plt.title(f'{joint_name.capitalize()} trajectory over time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(x_coords, y_coords, 'b-', alpha=0.7)
        plt.scatter(x_coords[0], y_coords[0], c='green', s=100, label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='End')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'{joint_name.capitalize()} 2D trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def run_full_test(self, video_files, calibration_file=None):
        """Run complete OpenCap test suite"""
        print("Starting OpenCap test suite...")
        
        # Test 1: Camera calibration
        if calibration_file and os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
            self.test_camera_calibration(calibration_data)
        else:
            print("No calibration file provided, skipping calibration test")
        
        # Test 2: Pose estimation
        pose_results = self.test_pose_estimation(video_files)
        
        # Test 3: Data validation
        for i, poses in enumerate(pose_results):
            print(f"\nValidating camera {i+1} data:")
            self.validate_pose_data(poses)
        
        # Test 4: Visualization
        if pose_results:
            self.plot_pose_trajectory(pose_results[0])
        
        print("\nOpenCap test suite completed!")

# Example usage
if __name__ == "__main__":
    tester = OpenCapTester()
    
    # Example video files (replace with actual paths)
    video_files = [
        "camera1_video.mp4",
        "camera2_video.mp4"
    ]
    
    tester.run_full_test(video_files, "calibration.json")
