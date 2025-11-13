import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

class YOLOVisualizer:
    def __init__(self, model_path=None, config_file=None):
        """Initialize YOLO visualizer"""
        self.model_path = model_path
        self.config = self.load_config(config_file)
        self.model = None
        self.class_names = self.load_class_names()
        self.colors = self.generate_colors()
        
    def load_config(self, config_file):
        """Load configuration"""
        default_config = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'input_size': 640,
            'max_detections': 100
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                default_config.update(config)
                
        return default_config
    
    def load_class_names(self):
        """Load COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def generate_colors(self):
        """Generate colors for different classes"""
        np.random.seed(42)
        return [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
                for _ in range(len(self.class_names))]
    
    def load_model(self):
        """Load YOLO model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
            else:
                # Load pretrained model
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            self.model.conf = self.config['confidence_threshold']
            self.model.iou = self.config['nms_threshold']
            self.model.max_det = self.config['max_detections']
            
            print("YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_objects(self, image):
        """Detect objects in image"""
        if self.model is None:
            if not self.load_model():
                return []
        
        # Run inference
        results = self.model(image)
        
        # Parse results
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= self.config['confidence_threshold']:
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def process_image(self, image_path):
        """Process single image"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None, []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None, []
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        detections = self.detect_objects(image_rgb)
        
        # Draw results
        result_image = self.draw_detections(image_rgb, detections)
        
        return result_image, detections
    
    def process_video(self, video_path, output_path=None):
        """Process video file"""
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects
            detections = self.detect_objects(frame_rgb)
            
            # Store detections with frame info
            frame_detections = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'detections': detections
            }
            all_detections.append(frame_detections)
            
            # Draw results
            result_frame = self.draw_detections(frame_rgb, detections)
            result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame if output specified
            if writer:
                writer.write(result_frame_bgr)
            
            # Display frame (optional)
            cv2.imshow('YOLO Detection', result_frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        return all_detections
    
    def analyze_detections(self, detections_list):
        """Analyze detection statistics"""
        if not detections_list:
            return {}
        
        # Count detections per class
        class_counts = {}
        confidence_scores = []
        
        for frame_data in detections_list:
            for detection in frame_data['detections']:
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                confidence_scores.append(confidence)
        
        # Calculate statistics
        stats = {
            'total_detections': sum(class_counts.values()),
            'unique_classes': len(class_counts),
            'class_counts': class_counts,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0
        }
        
        return stats
    
    def plot_detection_statistics(self, stats):
        """Plot detection statistics"""
        if not stats or 'class_counts' not in stats:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Class distribution
        classes = list(stats['class_counts'].keys())
        counts = list(stats['class_counts'].values())
        
        ax1.bar(classes, counts)
        ax1.set_title('Object Detection Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Summary statistics
        summary_data = [
            f"Total Detections: {stats['total_detections']}",
            f"Unique Classes: {stats['unique_classes']}",
            f"Avg Confidence: {stats['avg_confidence']:.3f}",
            f"Min Confidence: {stats['min_confidence']:.3f}",
            f"Max Confidence: {stats['max_confidence']:.3f}"
        ]
        
        ax2.text(0.1, 0.5, '\n'.join(summary_data), transform=ax2.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax2.set_title('Detection Statistics')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    visualizer = YOLOVisualizer()
    
    # Process single image
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        result_image, detections = visualizer.process_image(image_path)
        if result_image is not None:
            plt.figure(figsize=(12, 8))
            plt.imshow(result_image)
            plt.title(f'YOLO Detection Results - {len(detections)} objects detected')
            plt.axis('off')
            plt.show()
    
    # Process video
    video_path = "test_video.mp4"
    if os.path.exists(video_path):
        detections_list = visualizer.process_video(video_path, "output_video.mp4")
        stats = visualizer.analyze_detections(detections_list)
        visualizer.plot_detection_statistics(stats)
