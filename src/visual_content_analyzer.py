"""
Visual Content Analyzer using Vision-Language Models (VLM) to understand media content.
Analyzes images and videos to provide intelligent editing suggestions.
"""

import cv2
import numpy as np
from PIL import Image
import requests
import json
import base64
import io
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import tempfile

@dataclass
class ContentAnalysis:
    file_path: str
    content_type: str  # "image" or "video"
    dominant_colors: List[Tuple[int, int, int]]
    mood: str
    objects: List[str]
    scene_type: str  # "indoor", "outdoor", "portrait", "landscape", etc.
    lighting: str  # "bright", "dark", "natural", "artificial"
    movement: Optional[str] = None  # For videos: "static", "slow", "fast"
    text_detected: Optional[str] = None
    faces_count: int = 0
    quality_score: float = 0.0
    recommended_duration: float = 3.0
    style_tags: List[str] = None

    def __post_init__(self):
        if self.style_tags is None:
            self.style_tags = []

class VisualContentAnalyzer:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.vision_model = "llava"  # or "llava:13b" for better quality
        
        # Initialize computer vision components
        self.face_cascade = None
        self._load_cv_models()

    def _load_cv_models(self):
        """Load OpenCV models for basic analysis."""
        try:
            # Load face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"Warning: Could not load CV models: {e}")

    def analyze_media_file(self, file_path: str) -> ContentAnalysis:
        """
        Analyze a single media file (image or video).
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")
        
        # Determine content type
        content_type = self._get_content_type(file_path)
        
        if content_type == "image":
            return self._analyze_image(str(file_path))
        elif content_type == "video":
            return self._analyze_video(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def analyze_media_collection(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze a collection of media files and provide overall insights.
        """
        analyses = []
        
        for file_path in file_paths:
            try:
                analysis = self.analyze_media_file(file_path)
                analyses.append(analysis)
            except Exception as e:
                print(f"Warning: Failed to analyze {file_path}: {e}")
                continue
        
        if not analyses:
            return {"error": "No media files could be analyzed"}
        
        # Aggregate insights
        return self._aggregate_analyses(analyses)

    def _get_content_type(self, file_path: Path) -> str:
        """Determine if file is image or video."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        suffix = file_path.suffix.lower()
        
        if suffix in image_extensions:
            return "image"
        elif suffix in video_extensions:
            return "video"
        else:
            return "unknown"

    def _analyze_image(self, file_path: str) -> ContentAnalysis:
        """
        Analyze a single image file.
        """
        # Basic CV analysis
        basic_analysis = self._basic_image_analysis(file_path)
        
        # VLM analysis if available
        vlm_analysis = self._vlm_image_analysis(file_path)
        
        # Combine results
        return self._combine_analyses(file_path, "image", basic_analysis, vlm_analysis)

    def _analyze_video(self, file_path: str) -> ContentAnalysis:
        """
        Analyze a video file by extracting key frames.
        """
        # Extract key frame for analysis
        key_frame_path = self._extract_key_frame(file_path)
        
        if key_frame_path and os.path.exists(key_frame_path):
            try:
                # Analyze the key frame as an image
                frame_analysis = self._analyze_image(key_frame_path)
                
                # Add video-specific properties
                video_props = self._analyze_video_properties(file_path)
                
                # Update analysis with video properties
                frame_analysis.content_type = "video"
                frame_analysis.file_path = file_path  # Ensure correct file path
                frame_analysis.movement = video_props.get("movement", "static")
                frame_analysis.recommended_duration = video_props.get("duration", 3.0)
                
                return frame_analysis
                
            finally:
                # Clean up temporary frame
                try:
                    if os.path.exists(key_frame_path):
                        os.unlink(key_frame_path)
                except Exception as e:
                    print(f"Warning: Could not clean up temp file {key_frame_path}: {e}")
        
        # Fallback analysis for videos that can't be analyzed
        print(f"Using fallback analysis for video: {file_path}")
        return ContentAnalysis(
            file_path=file_path,
            content_type="video",
            dominant_colors=[(128, 128, 128)],
            mood="neutral",
            objects=[],
            scene_type="unknown",
            lighting="unknown",
            movement="unknown"
        )

    def _basic_image_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Perform basic computer vision analysis.
        """
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Get dominant colors
            dominant_colors = self._get_dominant_colors(image)
            
            # Detect faces
            faces_count = self._count_faces(image)
            
            # Analyze brightness/lighting
            lighting = self._analyze_lighting(image)
            
            # Calculate quality score (basic)
            quality_score = self._calculate_quality_score(image)
            
            return {
                "dominant_colors": dominant_colors,
                "faces_count": faces_count,
                "lighting": lighting,
                "quality_score": quality_score
            }
            
        except Exception as e:
            print(f"Basic analysis failed: {e}")
            return {"error": str(e)}

    def _vlm_image_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Use Vision-Language Model for advanced analysis.
        """
        if not self._is_vlm_available():
            return {}
        
        try:
            # Encode image to base64
            image_b64 = self._encode_image_to_base64(file_path)
            
            prompt = """
            Analyze this image and provide a JSON response with the following information:
            {
                "mood": "happy/sad/energetic/calm/dramatic/neutral",
                "objects": ["list of main objects/subjects in the image"],
                "scene_type": "indoor/outdoor/portrait/landscape/close-up/wide",
                "style_tags": ["aesthetic/modern/vintage/professional/casual/artistic"],
                "text_detected": "any text visible in the image or null",
                "recommended_duration": "number between 2-5 seconds based on content complexity"
            }
            
            Focus on elements that would be relevant for video editing decisions.
            """
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                vlm_output = result.get("response", "{}")
                return json.loads(vlm_output)
                
        except Exception as e:
            print(f"VLM analysis failed: {e}")
        
        return {}

    def _extract_key_frame(self, video_path: str) -> Optional[str]:
        """
        Extract a representative frame from video.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get middle frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = frame_count // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Save temporary frame with proper file handling
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                temp_file.close()  # Close the file handle before writing
                cv2.imwrite(temp_file.name, frame)
                
                # Verify file was created
                if os.path.exists(temp_file.name):
                    return temp_file.name
                
        except Exception as e:
            print(f"Key frame extraction failed: {e}")
        
        return None

    def _analyze_video_properties(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video-specific properties.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample a few frames to detect movement
            movement = self._detect_movement(cap)
            
            cap.release()
            
            # Recommend duration based on original video length
            recommended_duration = min(max(duration * 0.3, 2.0), 5.0)
            
            return {
                "duration": recommended_duration,
                "movement": movement,
                "original_duration": duration
            }
            
        except Exception as e:
            print(f"Video properties analysis failed: {e}")
            return {"duration": 3.0, "movement": "unknown"}

    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors using K-means clustering.
        """
        try:
            # Reshape image to be a list of pixels
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply K-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert to integers and return as tuples
            centers = np.uint8(centers)
            return [tuple(color[::-1]) for color in centers]  # BGR to RGB
            
        except Exception:
            return [(128, 128, 128)]  # Default gray

    def _count_faces(self, image: np.ndarray) -> int:
        """
        Count faces in image.
        """
        if self.face_cascade is None:
            return 0
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces)
        except Exception:
            return 0

    def _analyze_lighting(self, image: np.ndarray) -> str:
        """
        Analyze lighting conditions.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness > 180:
                return "bright"
            elif mean_brightness < 60:
                return "dark"
            else:
                return "natural"
        except Exception:
            return "unknown"

    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """
        Calculate basic quality score (0-1).
        """
        try:
            # Simple quality based on sharpness (Laplacian variance)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined threshold)
            quality = min(laplacian_var / 1000.0, 1.0)
            return quality
        except Exception:
            return 0.5

    def _detect_movement(self, cap) -> str:
        """
        Detect movement level in video.
        """
        try:
            # Sample a few frames and calculate optical flow
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = min(10, frame_count // 4)
            
            movements = []
            prev_gray = None
            
            for i in range(sample_frames):
                frame_pos = (i * frame_count) // sample_frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, None, None)
                    if flow[0] is not None:
                        movement_magnitude = np.mean(np.linalg.norm(flow[1], axis=1))
                        movements.append(movement_magnitude)
                
                prev_gray = gray
            
            if movements:
                avg_movement = np.mean(movements)
                if avg_movement > 5.0:
                    return "fast"
                elif avg_movement > 1.0:
                    return "slow"
                else:
                    return "static"
            
        except Exception:
            pass
        
        return "unknown"

    def _encode_image_to_base64(self, file_path: str) -> str:
        """
        Encode image to base64 for VLM API.
        """
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _combine_analyses(self, file_path: str, content_type: str, 
                         basic: Dict, vlm: Dict) -> ContentAnalysis:
        """
        Combine basic CV and VLM analyses.
        """
        return ContentAnalysis(
            file_path=file_path,
            content_type=content_type,
            dominant_colors=basic.get("dominant_colors", [(128, 128, 128)]),
            mood=vlm.get("mood", "neutral"),
            objects=vlm.get("objects", []),
            scene_type=vlm.get("scene_type", "unknown"),
            lighting=basic.get("lighting", "unknown"),
            text_detected=vlm.get("text_detected"),
            faces_count=basic.get("faces_count", 0),
            quality_score=basic.get("quality_score", 0.5),
            recommended_duration=float(vlm.get("recommended_duration", 3.0)),
            style_tags=vlm.get("style_tags", [])
        )

    def _aggregate_analyses(self, analyses: List[ContentAnalysis]) -> Dict[str, Any]:
        """
        Aggregate multiple content analyses.
        """
        if not analyses:
            return {}
        
        # Calculate overall metrics
        avg_quality = np.mean([a.quality_score for a in analyses])
        total_faces = sum(a.faces_count for a in analyses)
        
        # Most common mood
        moods = [a.mood for a in analyses if a.mood != "neutral"]
        overall_mood = max(set(moods), key=moods.count) if moods else "neutral"
        
        # Aggregate colors
        all_colors = []
        for analysis in analyses:
            all_colors.extend(analysis.dominant_colors)
        
        # Aggregate style tags
        all_style_tags = []
        for analysis in analyses:
            all_style_tags.extend(analysis.style_tags)
        
        style_frequency = {}
        for tag in all_style_tags:
            style_frequency[tag] = style_frequency.get(tag, 0) + 1
        
        return {
            "total_media_count": len(analyses),
            "average_quality": avg_quality,
            "total_faces": total_faces,
            "overall_mood": overall_mood,
            "dominant_colors": all_colors[:6],  # Top 6 colors
            "style_tags": sorted(style_frequency.keys(), key=style_frequency.get, reverse=True)[:5],
            "recommended_pacing": "fast" if overall_mood in ["energetic", "happy"] else "medium",
            "content_types": [a.content_type for a in analyses],
            "scene_types": [a.scene_type for a in analyses]
        }

    def _is_vlm_available(self) -> bool:
        """
        Check if VLM is available.
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any("llava" in model.get("name", "") for model in models)
        except:
            pass
        return False

    def suggest_edit_style(self, analyses: List[ContentAnalysis]) -> Dict[str, Any]:
        """
        Suggest editing style based on content analysis.
        """
        if not analyses:
            return {"style": "basic_slideshow"}
        
        # Analyze content characteristics
        has_faces = any(a.faces_count > 0 for a in analyses)
        quality_scores = [a.quality_score for a in analyses]
        avg_quality = np.mean(quality_scores)
        
        moods = [a.mood for a in analyses]
        energetic_content = sum(1 for mood in moods if mood in ["energetic", "happy"])
        
        # Generate suggestions
        suggestions = {
            "style": "basic_slideshow",
            "reasoning": []
        }
        
        if energetic_content > len(analyses) / 2:
            suggestions["style"] = "energetic"
            suggestions["reasoning"].append("Detected energetic content")
        
        if has_faces:
            suggestions["reasoning"].append("Portrait content detected - consider longer durations")
        
        if avg_quality > 0.7:
            suggestions["reasoning"].append("High quality content - can use dynamic transitions")
        
        return suggestions
