import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import numpy as np
from pathlib import Path
import tempfile

class VideoAnalyzer:
    def __init__(self):
        self.threshold = 30.0  # Threshold for scene detection
        
    def analyze_video(self, video_path):
        """
        Analyze the reference video to extract scene cuts and transitions.
        
        Args:
            video_path (str): Path to the reference video
            
        Returns:
            dict: Scene data including timestamps, transitions, and metadata
        """
        try:
            # Initialize scene detection
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))
            
            # Detect scenes
            scene_manager.detect_scenes(video)
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            
            # Extract video properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Analyze transitions between scenes
            scenes_data = []
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                duration = end_time - start_time
                
                # Determine transition type (simple cut by default)
                transition_type = "cut"
                if i > 0:
                    # Here you could analyze frames between scenes to detect fade/dissolve
                    # For now we'll just use cuts
                    pass
                
                scenes_data.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "transition": transition_type
                })
            
            return {
                "scenes": scenes_data,
                "metadata": {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "total_frames": total_frames,
                    "duration": total_frames / fps
                }
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing video: {str(e)}")
            
    def get_aspect_ratio(self, width, height):
        """Calculate the aspect ratio of the video."""
        return width / height if height != 0 else 0
        
    def resize_to_target(self, image, target_width, target_height):
        """Resize image to target dimensions while maintaining aspect ratio."""
        img_h, img_w = image.shape[:2]
        aspect = img_w / img_h
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect)
            
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to paste resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        # Paste resized image onto canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas