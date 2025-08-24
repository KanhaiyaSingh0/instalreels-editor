"""
Intelligent Video Engine that combines AI instruction parsing, visual analysis,
and advanced video generation techniques.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import random

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip
)
from moviepy.video.fx.all import (
    resize, fadein, fadeout, rotate, speedx
)
from moviepy.audio.fx.all import volumex

from .ai_instruction_parser import AIInstructionParser, EditInstruction, EditType, TransitionType
from .visual_content_analyzer import VisualContentAnalyzer, ContentAnalysis
from .audio_processor import AudioProcessor

class IntelligentVideoEngine:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.instruction_parser = AIInstructionParser(ollama_host)
        self.content_analyzer = VisualContentAnalyzer(ollama_host)
        self.audio_processor = AudioProcessor()
        
        self.target_size = (1080, 1920)  # Instagram Reels aspect ratio
        self.min_clip_duration = 1.0
        self.max_clip_duration = 6.0

    def create_intelligent_video(
        self,
        user_instruction: str,
        media_files: List[str],
        audio_path: Optional[str] = None,
        reference_video: Optional[str] = None,
        temp_dir: Optional[str] = None
    ) -> str:
        """
        Create a video based on natural language instructions and intelligent content analysis.
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        try:
            print(f"🎬 Processing instruction: {user_instruction}")
            
            # Step 1: Parse user instruction
            edit_instruction = self.instruction_parser.parse_instruction(
                user_instruction, len(media_files)
            )
            print(f"📋 Parsed instruction: {edit_instruction.edit_type.value}")
            
            # Step 2: Analyze media content
            print("🔍 Analyzing media content...")
            content_analyses = []
            for media_file in media_files:
                try:
                    # Validate file exists
                    if not os.path.exists(media_file):
                        print(f"⚠️ Media file not found: {media_file}")
                        continue
                        
                    analysis = self.content_analyzer.analyze_media_file(media_file)
                    content_analyses.append(analysis)
                    print(f"✅ Analyzed: {os.path.basename(media_file)}")
                except Exception as e:
                    print(f"⚠️ Failed to analyze {media_file}: {e}")
                    # Create a fallback analysis for failed files
                    fallback_analysis = ContentAnalysis(
                        file_path=media_file,
                        content_type="image" if media_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) else "video",
                        dominant_colors=[(128, 128, 128)],
                        mood="neutral",
                        objects=[],
                        scene_type="unknown",
                        lighting="unknown"
                    )
                    content_analyses.append(fallback_analysis)
            
            if not content_analyses:
                raise Exception("No media files could be analyzed")
            
            # Step 3: Get aggregated insights
            content_insights = self.content_analyzer._aggregate_analyses(content_analyses)
            print(f"✨ Content insights: {content_insights.get('overall_mood', 'neutral')} mood detected")
            
            # Step 4: Analyze audio if provided
            audio_analysis = None
            if audio_path:
                print("🎵 Analyzing audio...")
                audio_analysis = self.audio_processor.detect_beats(audio_path)
            
            # Step 5: Generate optimized edit plan
            edit_plan = self._create_edit_plan(
                edit_instruction, content_analyses, content_insights, audio_analysis
            )
            print(f"📝 Generated edit plan with {len(edit_plan['clips'])} clips")
            
            # Step 6: Create video
            output_path = self._execute_edit_plan(
                edit_plan, media_files, audio_path, temp_dir
            )
            
            print(f"✅ Video created successfully: {output_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Error creating intelligent video: {str(e)}")

    def _create_edit_plan(
        self,
        instruction: EditInstruction,
        content_analyses: List[ContentAnalysis],
        content_insights: Dict[str, Any],
        audio_analysis: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create an intelligent edit plan based on instruction and content analysis.
        """
        plan = {
            "clips": [],
            "audio_sync": instruction.music_sync and audio_analysis is not None,
            "overall_duration": 0,
            "style": instruction.edit_type.value
        }
        
        # Determine clip ordering based on content
        ordered_analyses = self._optimize_clip_order(content_analyses, instruction)
        
        # Calculate timing
        if instruction.music_sync and audio_analysis:
            clip_timings = self._sync_to_beats(ordered_analyses, audio_analysis, instruction)
        else:
            clip_timings = self._calculate_standard_timings(ordered_analyses, instruction)
        
        # Create clip plans
        for i, (analysis, timing) in enumerate(zip(ordered_analyses, clip_timings)):
            clip_plan = self._create_clip_plan(
                analysis, timing, instruction, i, len(ordered_analyses)
            )
            plan["clips"].append(clip_plan)
            plan["overall_duration"] += timing["duration"]
        
        # Add global effects based on instruction
        plan["global_effects"] = self._plan_global_effects(instruction, content_insights)
        
        return plan

    def _optimize_clip_order(
        self, 
        analyses: List[ContentAnalysis], 
        instruction: EditInstruction
    ) -> List[ContentAnalysis]:
        """
        Optimize the order of clips based on content and instruction.
        """
        if instruction.edit_type == EditType.STORYTELLING:
            # Sort by quality and composition for storytelling
            return sorted(analyses, key=lambda a: (a.quality_score, a.faces_count), reverse=True)
        
        elif instruction.edit_type == EditType.ENERGETIC:
            # Alternate between different content types for energy
            faces = [a for a in analyses if a.faces_count > 0]
            no_faces = [a for a in analyses if a.faces_count == 0]
            
            result = []
            max_len = max(len(faces), len(no_faces))
            for i in range(max_len):
                if i < len(faces):
                    result.append(faces[i])
                if i < len(no_faces):
                    result.append(no_faces[i])
            
            return result
        
        elif instruction.edit_type == EditType.AESTHETIC:
            # Sort by visual appeal (quality + lighting)
            return sorted(analyses, 
                         key=lambda a: a.quality_score * (1.2 if a.lighting == "natural" else 1.0), 
                         reverse=True)
        
        else:
            # Default: maintain original order but prioritize high quality
            return sorted(analyses, key=lambda a: a.quality_score, reverse=True)

    def _sync_to_beats(
        self,
        analyses: List[ContentAnalysis],
        audio_analysis: Dict,
        instruction: EditInstruction
    ) -> List[Dict[str, float]]:
        """
        Synchronize clip timings to audio beats.
        """
        beat_times = audio_analysis.get("beat_times", [])
        if not beat_times:
            return self._calculate_standard_timings(analyses, instruction)
        
        timings = []
        current_time = 0
        
        for i, analysis in enumerate(analyses):
            # Find next suitable beat interval
            beat_duration = self._find_beat_interval(beat_times, current_time, instruction.pacing)
            
            # Adjust duration based on content
            content_duration = analysis.recommended_duration
            final_duration = self._blend_durations(beat_duration, content_duration, 0.7)  # 70% beat, 30% content
            
            timings.append({
                "start_time": current_time,
                "duration": final_duration,
                "beat_aligned": True
            })
            
            current_time += final_duration
        
        return timings

    def _find_beat_interval(self, beat_times: List[float], current_time: float, pacing: str) -> float:
        """
        Find appropriate beat interval based on pacing.
        """
        if not beat_times:
            return 3.0
        
        # Find beats around current time
        future_beats = [b for b in beat_times if b > current_time]
        
        if len(future_beats) < 2:
            return 3.0
        
        # Calculate intervals
        intervals = [future_beats[i+1] - future_beats[i] for i in range(len(future_beats)-1)]
        avg_interval = np.mean(intervals)
        
        # Adjust based on pacing
        if pacing == "fast":
            return avg_interval
        elif pacing == "slow":
            return avg_interval * 2
        else:
            return avg_interval * 1.5

    def _calculate_standard_timings(
        self,
        analyses: List[ContentAnalysis],
        instruction: EditInstruction
    ) -> List[Dict[str, float]]:
        """
        Calculate standard timings without beat sync.
        """
        timings = []
        current_time = 0
        
        for analysis in analyses:
            # Base duration from instruction
            base_duration = instruction.duration_per_clip
            
            # Adjust based on content analysis
            if analysis.faces_count > 0:
                base_duration *= 1.3  # Longer for faces
            
            if analysis.quality_score > 0.8:
                base_duration *= 1.2  # Longer for high quality
            
            if instruction.pacing == "fast":
                base_duration *= 0.8
            elif instruction.pacing == "slow":
                base_duration *= 1.5
            
            # Clamp to reasonable range
            final_duration = max(self.min_clip_duration, 
                               min(self.max_clip_duration, base_duration))
            
            timings.append({
                "start_time": current_time,
                "duration": final_duration,
                "beat_aligned": False
            })
            
            current_time += final_duration
        
        return timings

    def _create_clip_plan(
        self,
        analysis: ContentAnalysis,
        timing: Dict[str, float],
        instruction: EditInstruction,
        clip_index: int,
        total_clips: int
    ) -> Dict[str, Any]:
        """
        Create a detailed plan for a single clip.
        """
        return {
            "file_path": analysis.file_path,
            "content_type": analysis.content_type,
            "start_time": timing["start_time"],
            "duration": timing["duration"],
            "transitions": self._select_transitions(instruction, analysis, clip_index, total_clips),
            "effects": self._select_effects(instruction, analysis),
            "text_overlay": self._plan_text_overlay(instruction, analysis, clip_index),
            "audio_sync": timing.get("beat_aligned", False),
            "priority_score": analysis.quality_score
        }

    def _select_transitions(
        self,
        instruction: EditInstruction,
        analysis: ContentAnalysis,
        clip_index: int,
        total_clips: int
    ) -> Dict[str, Any]:
        """
        Select appropriate transitions for the clip.
        """
        transitions = {
            "in": None,
            "out": None
        }
        
        # Skip transitions for first and last clips in certain cases
        if clip_index == 0:
            transitions["in"] = "fadein" if instruction.edit_type != EditType.ENERGETIC else None
        
        if clip_index == total_clips - 1:
            transitions["out"] = "fadeout"
        
        # Select transition type based on instruction and content
        available_transitions = instruction.transitions
        
        if instruction.edit_type == EditType.ENERGETIC:
            # Prefer cuts and slides for energy
            preferred = [TransitionType.CUT, TransitionType.SLIDE]
            transitions["out"] = random.choice([t for t in available_transitions if t in preferred] or available_transitions)
        
        elif instruction.edit_type == EditType.CALM:
            # Prefer fades and dissolves for calm
            preferred = [TransitionType.FADE, TransitionType.DISSOLVE]
            transitions["out"] = random.choice([t for t in available_transitions if t in preferred] or available_transitions)
        
        else:
            # Mix transitions based on content
            if analysis.quality_score > 0.7:
                transitions["out"] = random.choice(available_transitions)
            else:
                transitions["out"] = TransitionType.FADE  # Safe default
        
        return transitions

    def _select_effects(
        self,
        instruction: EditInstruction,
        analysis: ContentAnalysis
    ) -> List[Dict[str, Any]]:
        """
        Select effects based on instruction and content analysis.
        """
        effects = []
        
        # Ken Burns effect for images with high quality
        if (analysis.content_type == "image" and 
            analysis.quality_score > 0.6 and 
            instruction.edit_type != EditType.BASIC_SLIDESHOW):
            effects.append({
                "type": "ken_burns",
                "intensity": 0.1 if instruction.pacing == "slow" else 0.05
            })
        
        # Speed adjustments for videos
        if analysis.content_type == "video":
            if instruction.edit_type == EditType.ENERGETIC:
                effects.append({"type": "speed", "factor": 1.2})
            elif instruction.pacing == "slow":
                effects.append({"type": "speed", "factor": 0.9})
        
        # Color grading based on mood
        if instruction.edit_type == EditType.AESTHETIC:
            effects.append({
                "type": "color_grade",
                "style": analysis.mood
            })
        
        return effects

    def _plan_text_overlay(
        self,
        instruction: EditInstruction,
        analysis: ContentAnalysis,
        clip_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Plan text overlay for the clip.
        """
        if not instruction.text_overlay:
            return None
        
        # Only add text to first clip or high-quality clips
        if clip_index == 0 or analysis.quality_score > 0.8:
            return {
                "text": instruction.text_overlay,
                "position": "center",
                "font_size": 60,
                "color": "white",
                "duration": min(3.0, analysis.recommended_duration * 0.8)
            }
        
        return None

    def _plan_global_effects(
        self,
        instruction: EditInstruction,
        content_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Plan global effects that apply to the entire video.
        """
        effects = []
        
        # Audio effects
        if instruction.edit_type == EditType.ENERGETIC:
            effects.append({
                "type": "audio_boost",
                "factor": 1.1
            })
        
        # Color consistency
        if instruction.edit_type == EditType.AESTHETIC:
            effects.append({
                "type": "color_consistency",
                "target_mood": content_insights.get("overall_mood", "neutral")
            })
        
        return effects

    def _execute_edit_plan(
        self,
        edit_plan: Dict[str, Any],
        media_files: List[str],
        audio_path: Optional[str],
        temp_dir: str
    ) -> str:
        """
        Execute the edit plan and create the final video.
        """
        clips = []
        
        # Create individual clips
        for clip_plan in edit_plan["clips"]:
            try:
                clip = self._create_intelligent_clip(clip_plan)
                if clip:
                    clips.append(clip)
            except Exception as e:
                print(f"⚠️ Failed to create clip {clip_plan['file_path']}: {e}")
                continue
        
        if not clips:
            raise Exception("No valid clips were created")
        
        # Concatenate clips
        final_video = concatenate_videoclips(clips)
        
        # Apply global effects
        final_video = self._apply_global_effects(final_video, edit_plan["global_effects"])
        
        # Add audio
        if audio_path:
            final_video = self._add_intelligent_audio(final_video, audio_path)
        
        # Generate output
        output_path = os.path.join("output", "ai_generated_video.mp4")
        
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(temp_dir, "temp-audio.m4a"),
            remove_temp=True,
            fps=30,
            verbose=False,
            logger=None
        )
        
        return output_path

    def _create_intelligent_clip(self, clip_plan: Dict[str, Any]) -> Optional[Any]:
        """
        Create a single intelligent clip based on the plan.
        """
        file_path = clip_plan["file_path"]
        duration = clip_plan["duration"]
        
        try:
            # Create base clip
            if clip_plan["content_type"] == "image":
                clip = self._create_intelligent_image_clip(file_path, duration)
            else:
                clip = self._create_intelligent_video_clip(file_path, duration)
            
            # Apply effects
            for effect in clip_plan["effects"]:
                clip = self._apply_effect(clip, effect)
            
            # Apply transitions
            clip = self._apply_transitions(clip, clip_plan["transitions"])
            
            # Add text overlay
            if clip_plan["text_overlay"]:
                clip = self._add_text_overlay(clip, clip_plan["text_overlay"])
            
            return clip
            
        except Exception as e:
            print(f"Error creating intelligent clip: {e}")
            return None

    def _create_intelligent_image_clip(self, image_path: str, duration: float) -> Any:
        """
        Create an intelligent image clip with automatic enhancements.
        """
        from PIL import Image
        
        # Load and process image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Create clip
        clip = ImageClip(img_array, duration=duration)
        clip = clip.resize(self.target_size)
        
        return clip

    def _create_intelligent_video_clip(self, video_path: str, duration: float) -> Any:
        """
        Create an intelligent video clip with smart trimming.
        """
        clip = VideoFileClip(video_path)
        
        if clip.duration <= 0:
            raise Exception("Invalid video duration")
        
        # Smart trimming - use the best part of the video
        if clip.duration > duration:
            # Use middle section by default, but could be enhanced with motion detection
            start_time = (clip.duration - duration) / 2
            clip = clip.subclip(start_time, start_time + duration)
        elif clip.duration < duration:
            # Loop if too short
            loops_needed = int(np.ceil(duration / clip.duration))
            clip = concatenate_videoclips([clip] * loops_needed).subclip(0, duration)
        
        clip = clip.resize(self.target_size)
        return clip

    def _apply_effect(self, clip: Any, effect: Dict[str, Any]) -> Any:
        """
        Apply a single effect to a clip.
        """
        effect_type = effect["type"]
        
        try:
            if effect_type == "ken_burns":
                # Simple ken burns effect (zoom)
                intensity = effect.get("intensity", 0.05)
                start_size = (1.0, 1.0)
                end_size = (1.0 + intensity, 1.0 + intensity)
                
                return clip.resize(lambda t: [
                    start_size[0] + (end_size[0] - start_size[0]) * t / clip.duration,
                    start_size[1] + (end_size[1] - start_size[1]) * t / clip.duration
                ])
            
            elif effect_type == "speed":
                factor = effect.get("factor", 1.0)
                return clip.fx(speedx, factor)
            
            elif effect_type == "color_grade":
                # Basic color grading - could be enhanced
                style = effect.get("style", "neutral")
                if style == "energetic":
                    return clip.fx(lambda c: c.fl_image(lambda img: np.clip(img * 1.1, 0, 255)))
                elif style == "calm":
                    return clip.fx(lambda c: c.fl_image(lambda img: np.clip(img * 0.9, 0, 255)))
            
        except Exception as e:
            print(f"Failed to apply effect {effect_type}: {e}")
        
        return clip

    def _apply_transitions(self, clip: Any, transitions: Dict[str, Any]) -> Any:
        """
        Apply transitions to a clip.
        """
        try:
            if transitions.get("in") == TransitionType.FADEIN or transitions.get("in") == "fadein":
                clip = clip.fx(fadein, 0.5)
            
            if transitions.get("out") == TransitionType.FADEOUT or transitions.get("out") == "fadeout":
                clip = clip.fx(fadeout, 0.5)
            
            # For other transition types, we'll use fadeout as fallback
            if transitions.get("out") in [TransitionType.FADE, TransitionType.DISSOLVE]:
                clip = clip.fx(fadeout, 0.3)
            
        except Exception as e:
            print(f"Failed to apply transitions: {e}")
        
        return clip

    def _add_text_overlay(self, clip: Any, text_config: Dict[str, Any]) -> Any:
        """
        Add text overlay to a clip.
        """
        try:
            text_clip = TextClip(
                text_config["text"],
                fontsize=text_config.get("font_size", 60),
                color=text_config.get("color", "white"),
                font="Arial-Bold"
            ).set_duration(text_config.get("duration", 3.0))
            
            # Position text
            position = text_config.get("position", "center")
            if position == "center":
                text_clip = text_clip.set_position("center")
            
            return CompositeVideoClip([clip, text_clip])
            
        except Exception as e:
            print(f"Failed to add text overlay: {e}")
            return clip

    def _apply_global_effects(self, video: Any, effects: List[Dict[str, Any]]) -> Any:
        """
        Apply global effects to the entire video.
        """
        for effect in effects:
            try:
                if effect["type"] == "audio_boost":
                    factor = effect.get("factor", 1.1)
                    if video.audio:
                        video = video.set_audio(video.audio.fx(volumex, factor))
                
                # Add more global effects as needed
                
            except Exception as e:
                print(f"Failed to apply global effect: {e}")
        
        return video

    def _add_intelligent_audio(self, video: Any, audio_path: str) -> Any:
        """
        Add audio with intelligent mixing.
        """
        try:
            audio = AudioFileClip(audio_path)
            
            # Match audio duration to video
            if audio.duration > video.duration:
                audio = audio.subclip(0, video.duration)
            else:
                # Loop audio if needed
                loops_needed = int(np.ceil(video.duration / audio.duration))
                audio = concatenate_audioclips([audio] * loops_needed).subclip(0, video.duration)
            
            return video.set_audio(audio)
            
        except Exception as e:
            print(f"Failed to add audio: {e}")
            return video

    def _blend_durations(self, duration1: float, duration2: float, weight1: float) -> float:
        """
        Blend two durations with given weight.
        """
        return duration1 * weight1 + duration2 * (1 - weight1)
