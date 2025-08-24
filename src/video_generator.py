from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.video.fx.all import resize, fadein, fadeout
import numpy as np
from pathlib import Path
import tempfile
import cv2
from PIL import Image
import os

class VideoGenerator:
    def __init__(self):
        self.transition_duration = 60  # Duration for transitions in seconds
        self.target_size = (1080, 1920)  # Instagram Reels aspect ratio (9:16)
        
    def create_video(self, scene_data, beat_data, media_files, audio_path, temp_dir):
        """
        Create the final video using the analyzed scene data and provided media.
        
        Args:
            scene_data (dict): Scene timing and transition data
            beat_data (dict): Beat timestamps and tempo information
            media_files (list): List of media file paths
            audio_path (str): Path to the audio file
            temp_dir (str): Temporary directory for processing
            
        Returns:
            str: Path to the generated video
        """
        try:
            # Validate inputs
            if not media_files:
                raise Exception("No media files provided")
            
            # Prepare clips
            clips = []
            media_index = 0
            scenes = scene_data.get("scenes", [])
            
            # If no scenes detected, create a fallback using all media files with default durations
            if not scenes:
                print("No scenes detected, creating fallback video with default durations")
                default_duration = 3.0  # 3 seconds per media item
                for media_path in media_files:
                    try:
                        # Create clip based on media type
                        if self._is_image(media_path):
                            clip = self._create_image_clip(media_path, default_duration)
                        else:
                            clip = self._create_video_clip(media_path, default_duration)
                        
                        # Apply fade transition for better visual flow
                        clip = self._apply_fade_transition(clip)
                        clips.append(clip)
                    except Exception as e:
                        print(f"Warning: Failed to process media file {media_path}: {str(e)}")
                        continue
            else:
                # Use detected scenes
                for scene in scenes:
                    if media_index >= len(media_files):
                        media_index = 0  # Loop back to start if we run out of media
                    
                    media_path = media_files[media_index]
                    duration = scene["duration"]
                    
                    try:
                        # Create clip based on media type
                        if self._is_image(media_path):
                            clip = self._create_image_clip(media_path, duration)
                        else:
                            clip = self._create_video_clip(media_path, duration)
                        
                        # Apply transitions
                        if scene["transition"] == "fade":
                            clip = self._apply_fade_transition(clip)
                        
                        clips.append(clip)
                    except Exception as e:
                        print(f"Warning: Failed to process media file {media_path}: {str(e)}")
                        # Continue with next media file
                    
                    media_index += 1
            
            # Validate clips before concatenation
            if not clips:
                raise Exception("No valid clips were created")
            
            # Concatenate all clips
            final_video = concatenate_videoclips(clips)
            
            # Add audio
            audio = AudioFileClip(audio_path)

            # Trim or loop audio to match video duration
            if audio.duration > final_video.duration:
                audio = audio.subclip(0, final_video.duration)
            else:
                # Loop audio if it's shorter than video
                repeats = int(np.ceil(final_video.duration / audio.duration))
                audio = concatenate_audioclips([audio] * repeats).subclip(0, final_video.duration)
            
            final_video = final_video.set_audio(audio)
            
            # Generate output path
            output_path = os.path.join("output", "generated_video.mp4")
            
            # Write final video
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(temp_dir, "temp-audio.m4a"),
                remove_temp=True,
                fps=30
            )
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating video: {str(e)}")
    
    def _is_image(self, file_path):
        """Check if file is an image based on extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        return Path(file_path).suffix.lower() in image_extensions
    
    def _create_image_clip(self, image_path, duration):
        """Create a video clip from an image."""
        try:
            # Load and resize image
            img = Image.open(image_path)
            
            # Handle different image modes
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                # Create a white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            img = np.array(img)
            
            # Create clip
            clip = ImageClip(img, duration=duration)
            
            # Resize to target size
            clip = clip.resize(self.target_size)
            
            return clip
            
        except Exception as e:
            raise Exception(f"Error creating image clip from {image_path}: {str(e)}")
    
    def _create_video_clip(self, video_path, duration):
        """Create a video clip from a video file."""
        try:
            clip = VideoFileClip(video_path)
            
            # Ensure clip has a valid duration
            if clip.duration is None or clip.duration <= 0:
                raise Exception(f"Invalid video duration for {video_path}")
            
            # Trim to desired duration
            if clip.duration > duration:
                clip = clip.subclip(0, duration)
            else:
                # If video is shorter than desired duration, loop it
                loops_needed = int(np.ceil(duration / clip.duration))
                if loops_needed > 1:
                    clip = concatenate_videoclips([clip] * loops_needed).subclip(0, duration)
            
            # Resize to target size
            clip = clip.resize(self.target_size)
            
            return clip
            
        except Exception as e:
            raise Exception(f"Error creating video clip from {video_path}: {str(e)}")
    
    def _apply_fade_transition(self, clip, fade_duration=0.5):
        """Apply fade in/out transition to a clip."""
        try:
            return clip.fx(fadein, fade_duration).fx(fadeout, fade_duration)
        except Exception:
            return clip
    
    def _sync_to_beats(self, clip, beat_times):
        """
        Synchronize clip transitions with beat times.
        This is a simple implementation - could be enhanced for better beat matching.
        """
        # Find the closest beat to the clip's start/end
        clip_start = min(beat_times, key=lambda x: abs(x - clip.start))
        clip_end = min(beat_times, key=lambda x: abs(x - clip.end))
        
        # Adjust clip timing
        return clip.subclip(clip_start, clip_end)