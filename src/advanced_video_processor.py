"""
Advanced Video Processor with Automatic Lighting, Intelligent Cuts, and Multiple Output Options
"""

import cv2
import numpy as np
from moviepy.editor import *
from moviepy.video.fx.all import *
from moviepy.audio.fx.all import *
import librosa
from typing import List, Dict, Optional, Tuple
import random
import os

class AdvancedVideoProcessor:
    def __init__(self):
        self.target_size = (1080, 1920)  # Instagram Reels format
        self.lighting_presets = {
            "cinematic": {"brightness": 1.1, "contrast": 1.2, "saturation": 0.9, "warmth": 1.1},
            "vibrant": {"brightness": 1.2, "contrast": 1.3, "saturation": 1.4, "warmth": 1.0},
            "moody": {"brightness": 0.8, "contrast": 1.4, "saturation": 0.7, "warmth": 0.9},
            "natural": {"brightness": 1.0, "contrast": 1.1, "saturation": 1.0, "warmth": 1.0},
            "dramatic": {"brightness": 0.9, "contrast": 1.5, "saturation": 0.8, "warmth": 1.2}
        }
        
    def apply_lighting_effects(self, clip, preset_name: str = "cinematic"):
        """Apply automatic lighting corrections and effects."""
        preset = self.lighting_presets.get(preset_name, self.lighting_presets["natural"])
        
        def lighting_frame(frame):
            # Convert to float for processing
            frame = frame.astype(np.float32) / 255.0
            
            # Apply brightness
            frame = frame * preset["brightness"]
            
            # Apply contrast
            frame = (frame - 0.5) * preset["contrast"] + 0.5
            
            # Apply saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] *= preset["saturation"]
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Apply warmth (color temperature)
            if preset["warmth"] > 1.0:
                # Warmer (more orange/red)
                frame[:, :, 0] *= preset["warmth"]  # Red channel
                frame[:, :, 2] *= (2 - preset["warmth"])  # Blue channel
            else:
                # Cooler (more blue)
                frame[:, :, 0] *= preset["warmth"]  # Red channel
                frame[:, :, 2] *= (2 - preset["warmth"])  # Blue channel
            
            # Clamp values
            frame = np.clip(frame, 0, 1)
            
            # Convert back to uint8
            return (frame * 255).astype(np.uint8)
        
        return clip.fl_image(lighting_frame)
    
    def analyze_reference_video(self, reference_video_path: str) -> Dict:
        """Analyze reference video to extract timing, style, and structure."""
        try:
            from moviepy.editor import VideoFileClip
            ref_video = VideoFileClip(reference_video_path)
            
            # Extract basic info
            duration = ref_video.duration
            fps = ref_video.fps
            
            # For scene-based replacement, we'll split into 2 parts (human and tiger)
            # This is a simplified approach - in a real implementation, you'd use scene detection
            scene_durations = [duration / 2, duration / 2]  # Split into two equal parts
            
            # Extract audio for beat analysis
            if ref_video.audio is not None:
                # Save audio temporarily for analysis
                temp_audio_path = "temp_reference_audio.wav"
                ref_video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                ref_beats = self.detect_music_beats(temp_audio_path)
                os.remove(temp_audio_path)
            else:
                ref_beats = []
            
            print(f"🎬 Reference video analyzed: {duration:.1f}s, {fps} fps, {len(ref_beats)} beats")
            print(f"🎬 Split into 2 scenes: {scene_durations[0]:.1f}s (human part), {scene_durations[1]:.1f}s (tiger part)")
            
            return {
                "duration": duration,
                "fps": fps,
                "scene_durations": scene_durations,
                "beats": ref_beats,
                "total_scenes": 2,
                "video_path": reference_video_path
            }
            
        except Exception as e:
            print(f"⚠️ Reference video analysis failed: {str(e)}")
            return None
    
    def detect_music_beats(self, audio_path: str) -> List[float]:
        """Detect beat timestamps from music file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Detect beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Convert beat frames to timestamps
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Convert to list and handle numpy array formatting
            beat_list = []
            for beat in beat_times:
                try:
                    beat_list.append(float(beat))
                except (ValueError, TypeError):
                    continue
            
            print(f"🎵 Detected {len(beat_list)} beats at {tempo:.1f} BPM")
            return beat_list
            
        except Exception as e:
            print(f"⚠️ Beat detection failed: {str(e)}")
            return []
    
    def create_intelligent_cuts(self, clips: List, beat_times: List[float], 
                              target_duration: Optional[float] = None, 
                              reference_timing: Optional[Dict] = None) -> List:
        """Create intelligent cuts based on music beats and reference timing."""
        if not beat_times:
            return clips
        
        intelligent_clips = []
        current_time = 0
        
        for i, clip in enumerate(clips):
            # Find the closest beat to current time
            if beat_times:
                closest_beat = min(beat_times, key=lambda x: abs(x - current_time))
                # Adjust clip start to beat
                if abs(closest_beat - current_time) < 0.5:  # Within 0.5 seconds
                    current_time = closest_beat
            
            # Determine clip duration based on beats
            if len(beat_times) > 1:
                # Find next beat for duration
                next_beats = [b for b in beat_times if b > current_time]
                if next_beats:
                    duration = next_beats[0] - current_time
                    # Ensure reasonable duration
                    duration = max(1.0, min(duration, 4.0))
                else:
                    duration = 2.0
            else:
                duration = 2.0
            
            # Apply reference timing if available
            if reference_timing and reference_timing.get("scene_durations"):
                # Use reference video scene durations
                if i < len(reference_timing["scene_durations"]):
                    duration = reference_timing["scene_durations"][i]
                else:
                    # If we have more clips than reference scenes, use average
                    avg_duration = sum(reference_timing["scene_durations"]) / len(reference_timing["scene_durations"])
                    duration = avg_duration
            elif target_duration:
                # Scale duration based on target
                duration = duration * (target_duration / sum(c.duration for c in clips))
            
            # Trim or extend clip to match duration
            if clip.duration > duration:
                clip = clip.subclip(0, duration)
            elif clip.duration < duration:
                # Loop clip to fill duration
                loops_needed = int(np.ceil(duration / clip.duration))
                clip = concatenate_videoclips([clip] * loops_needed).subclip(0, duration)
            
            intelligent_clips.append(clip)
            current_time += duration
        
        return intelligent_clips
    
    def generate_multiple_versions(self, base_clips: List, audio_path: str, 
                                 instruction: str, target_duration: float = 30.0, 
                                 reference_video_path: str = None) -> Dict[str, VideoFileClip]:
        """Generate 3 different versions of the video for user selection."""
        
        versions = {}
        
        # Analyze reference video for timing and style
        reference_timing = self.analyze_reference_video(reference_video_path) if reference_video_path else None
        
        # Version 1: Energetic/Fast-paced
        print("🎬 Creating Version 1: Energetic & Fast-paced")
        energetic_clips = []
        for clip in base_clips:
            # Apply speed effect
            fast_clip = clip.fx(speedx, 1.3)
            # Apply vibrant lighting
            fast_clip = self.apply_lighting_effects(fast_clip, "vibrant")
            energetic_clips.append(fast_clip)
        
        # Sync to beats and reference timing
        beat_times = self.detect_music_beats(audio_path)
        energetic_clips = self.create_intelligent_cuts(energetic_clips, beat_times, target_duration, reference_timing)
        
        final_energetic = concatenate_videoclips(energetic_clips)
        # Adjust to target duration
        if final_energetic.duration > target_duration:
            final_energetic = final_energetic.subclip(0, target_duration)
        elif final_energetic.duration < target_duration:
            # Loop to fill duration
            loops = int(np.ceil(target_duration / final_energetic.duration))
            final_energetic = concatenate_videoclips([final_energetic] * loops).subclip(0, target_duration)
        
        versions["energetic"] = final_energetic
        
        # Version 2: Cinematic/Dramatic
        print("🎬 Creating Version 2: Cinematic & Dramatic")
        cinematic_clips = []
        for clip in base_clips:
            # Apply cinematic lighting
            cinematic_clip = self.apply_lighting_effects(clip, "cinematic")
            # Add subtle zoom effect
            cinematic_clip = cinematic_clip.resize(1.1)
            cinematic_clips.append(cinematic_clip)
        
        # Slower, more dramatic timing
        cinematic_clips = self.create_intelligent_cuts(cinematic_clips, beat_times[::2] if beat_times else [], target_duration, reference_timing)
        
        final_cinematic = concatenate_videoclips(cinematic_clips)
        # Adjust to target duration
        if final_cinematic.duration > target_duration:
            final_cinematic = final_cinematic.subclip(0, target_duration)
        elif final_cinematic.duration < target_duration:
            # Loop to fill duration
            loops = int(np.ceil(target_duration / final_cinematic.duration))
            final_cinematic = concatenate_videoclips([final_cinematic] * loops).subclip(0, target_duration)
        
        versions["cinematic"] = final_cinematic
        
        # Version 3: Natural/Smooth
        print("🎬 Creating Version 3: Natural & Smooth")
        natural_clips = []
        for clip in base_clips:
            # Apply natural lighting
            natural_clip = self.apply_lighting_effects(clip, "natural")
            # Add smooth transitions
            natural_clip = natural_clip.fx(fadein, 0.3).fx(fadeout, 0.3)
            natural_clips.append(natural_clip)
        
        # Smooth timing with longer clips
        natural_clips = self.create_intelligent_cuts(natural_clips, beat_times[::3] if beat_times else [], target_duration, reference_timing)
        
        final_natural = concatenate_videoclips(natural_clips)
        # Adjust to target duration
        if final_natural.duration > target_duration:
            final_natural = final_natural.subclip(0, target_duration)
        elif final_natural.duration < target_duration:
            # Loop to fill duration
            loops = int(np.ceil(target_duration / final_natural.duration))
            final_natural = concatenate_videoclips([final_natural] * loops).subclip(0, target_duration)
        
        versions["natural"] = final_natural
        
        return versions
    
    def create_scene_replacement_video(self, user_video_path: str, reference_video_path: str, 
                                     audio_path: str, instruction: str = "") -> VideoFileClip:
        """Create a video by replacing the first scene (human part) of reference video with user video."""
        try:
            from moviepy.editor import VideoFileClip
            
            print("🎬 Creating scene replacement video...")
            
            # Analyze reference video
            ref_analysis = self.analyze_reference_video(reference_video_path)
            if not ref_analysis:
                raise Exception("Failed to analyze reference video")
            
            # Load reference video
            ref_video = VideoFileClip(reference_video_path)
            
            # Load user video
            user_video = VideoFileClip(user_video_path)
            
            # Get scene durations
            human_part_duration = ref_analysis["scene_durations"][0]  # First half (human part)
            tiger_part_duration = ref_analysis["scene_durations"][1]  # Second half (tiger part)
            
            print(f"🎬 Human part duration: {human_part_duration:.1f}s")
            print(f"🎬 Tiger part duration: {tiger_part_duration:.1f}s")
            
            # Extract the tiger part from reference video (second half)
            tiger_part = ref_video.subclip(human_part_duration, ref_video.duration)
            
            # Process user video to match the human part duration
            if user_video.duration > human_part_duration:
                # Trim user video to fit
                user_video = user_video.subclip(0, human_part_duration)
            elif user_video.duration < human_part_duration:
                # Loop user video to fill the duration
                loops_needed = int(np.ceil(human_part_duration / user_video.duration))
                user_video = concatenate_videoclips([user_video] * loops_needed).subclip(0, human_part_duration)
            
            # Resize user video to match reference video dimensions
            user_video = user_video.resize(ref_video.size)
            
            # Apply lighting effects to user video to match reference style
            user_video = self.apply_lighting_effects(user_video, "natural")
            
            # Concatenate: user video (human part) + tiger part
            final_video = concatenate_videoclips([user_video, tiger_part])
            
            print(f"✅ Scene replacement video created: {final_video.duration:.1f}s")
            
            return final_video
            
        except Exception as e:
            print(f"❌ Scene replacement failed: {str(e)}")
            raise e
    
    def add_advanced_effects(self, clip, effect_type: str):
        """Add advanced visual effects."""
        if effect_type == "glitch":
            # Add glitch effect
            def glitch_frame(frame):
                if random.random() < 0.1:  # 10% chance of glitch
                    # Random color shift
                    frame = np.roll(frame, random.randint(-10, 10), axis=1)
                return frame
            return clip.fl_image(glitch_frame)
        
        elif effect_type == "vintage":
            # Add vintage effect
            def vintage_frame(frame):
                # Sepia tone
                frame = frame.astype(np.float32)
                frame = frame * [0.393, 0.769, 0.189]
                frame = np.clip(frame, 0, 255)
                return frame.astype(np.uint8)
            return clip.fl_image(vintage_frame)
        
        elif effect_type == "neon":
            # Add neon glow effect
            def neon_frame(frame):
                # Increase saturation and add glow
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] *= 1.5  # Increase saturation
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                return frame
            return clip.fl_image(neon_frame)
        
        return clip
    
    def create_final_video(self, version_clips: Dict[str, VideoFileClip], 
                          audio_path: str, output_dir: str = "output") -> Dict[str, str]:
        """Create final video files for all versions."""
        
        output_paths = {}
        
        for version_name, video_clip in version_clips.items():
            print(f"🎬 Rendering {version_name} version...")
            
            # Add audio
            if audio_path and os.path.exists(audio_path):
                audio = AudioFileClip(audio_path)
                # Sync audio to video duration
                if audio.duration > video_clip.duration:
                    audio = audio.subclip(0, video_clip.duration)
                else:
                    # Loop audio if needed
                    loops = int(np.ceil(video_clip.duration / audio.duration))
                    audio = concatenate_audioclips([audio] * loops).subclip(0, video_clip.duration)
                
                video_clip = video_clip.set_audio(audio)
            
            # Generate output path
            output_path = os.path.join(output_dir, f"nvidia_ai_video_{version_name}.mp4")
            
            # Render video
            video_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=30,
                verbose=False,
                logger=None
            )
            
            output_paths[version_name] = output_path
            print(f"✅ {version_name} version saved: {output_path}")
        
        return output_paths

# Example usage
if __name__ == "__main__":
    processor = AdvancedVideoProcessor()
    print("🚀 Advanced Video Processor ready!")
    print("Features:")
    print("• Automatic lighting corrections")
    print("• Intelligent beat-synchronized cuts")
    print("• Multiple output versions")
    print("• Advanced visual effects")
