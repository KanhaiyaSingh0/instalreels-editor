import gradio as gr
from src.video_analyzer import VideoAnalyzer
from src.audio_processor import AudioProcessor
from src.video_generator import VideoGenerator
from src.nvidia_cloud_ai_engine import NVIDIACloudVideoEngine, check_nvidia_setup
from src.advanced_video_processor import AdvancedVideoProcessor
from moviepy.editor import VideoFileClip, ImageClip
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

class EnhancedInstalreelsEditor:
    def __init__(self):
        self.video_analyzer = VideoAnalyzer()
        self.audio_processor = AudioProcessor()
        self.video_generator = VideoGenerator()
        self.advanced_processor = AdvancedVideoProcessor()
        
        # Load configuration from nvidia_config.env
        if os.path.exists("nvidia_config.env"):
            load_dotenv("nvidia_config.env")
        
        # Initialize NVIDIA Cloud AI Engine
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.nvidia_enabled = bool(nvidia_api_key and nvidia_api_key != "nvapi-your-key-here")
        
        if self.nvidia_enabled:
            try:
                self.nvidia_engine = NVIDIACloudVideoEngine(nvidia_api_key)
                print("🌟 NVIDIA Cloud AI enabled - pure cloud processing!")
            except Exception as e:
                print(f"⚠️ NVIDIA setup failed: {e}")
                self.nvidia_enabled = False
        else:
            print("⚡ NVIDIA API key not found - using traditional mode")
            print("🔧 Set NVIDIA_API_KEY to enable cloud AI features")
        
    def process_video_with_options(
        self,
        reference_video,
        media_files,
        music_file=None,
        user_instruction="",
        generate_multiple_versions=True,
        target_duration=30.0,
        music_choice="Use Reference Video Music",
        progress=gr.Progress()
    ):
        """
        Process video with multiple output options and advanced features.
        """
        try:
            # Normalize inputs to file path strings
            def _ensure_path(value):
                if value is None:
                    return None
                if isinstance(value, str):
                    return value
                if isinstance(value, dict) and "path" in value:
                    return value["path"]
                return value

            reference_video = _ensure_path(reference_video)
            music_file = _ensure_path(music_file)
            if isinstance(media_files, dict):
                media_files = [_ensure_path(media_files)]
            elif isinstance(media_files, list):
                media_files = [_ensure_path(m) for m in media_files]

            # Basic validation
            if not reference_video or not os.path.isfile(reference_video):
                raise gr.Error("Please upload a reference video. This will be used as the template for your video.")
            media_files = [m for m in (media_files or []) if isinstance(m, str) and os.path.isfile(m)]
            if not media_files:
                raise gr.Error("Please upload at least one photo or video in 'Your Photos/Videos'.")
            
            # Handle music choice
            if music_choice == "Use My Own Music":
                if not music_file or not os.path.isfile(music_file):
                    raise gr.Error("Please upload your music file when 'Use My Own Music' is selected.")
                final_music_file = music_file
            else:
                # Use reference video music
                final_music_file = reference_video

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                
                if generate_multiple_versions:
                    # Generate multiple versions with advanced features
                    progress(0.1, desc="🎬 Creating multiple video versions...")
                    
                    # Create base clips
                    base_clips = []
                    for media_file in media_files:
                        try:
                            if media_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                                # Image file
                                from PIL import Image
                                import numpy as np
                                img = Image.open(media_file).convert('RGB')
                                img_array = np.array(img)
                                clip = ImageClip(img_array, duration=3.0)
                            else:
                                # Video file
                                from moviepy.editor import VideoFileClip
                                clip = VideoFileClip(media_file)
                                if clip.duration > 6.0:
                                    clip = clip.subclip(0, 6.0)
                            
                            # Resize to target
                            clip = clip.resize(self.advanced_processor.target_size)
                            base_clips.append(clip)
                            
                        except Exception as e:
                            print(f"Warning: Failed to process {media_file}: {e}")
                            continue
                    
                    if not base_clips:
                        raise gr.Error("No valid media files could be processed.")
                    
                    progress(0.3, desc="🎨 Applying advanced lighting and effects...")
                    
                    # Check if user wants scene replacement
                    scene_replacement_keywords = ["replace", "human part", "tiger part", "scene replacement", "replace human", "replace first part"]
                    wants_scene_replacement = any(keyword in user_instruction.lower() for keyword in scene_replacement_keywords)
                    
                    if wants_scene_replacement and len(media_files) == 1:
                        # Scene replacement mode
                        progress(0.3, desc="🎬 Analyzing reference video for scene replacement...")
                        
                        try:
                            # Get the user video (should be the first media file)
                            user_video_path = media_files[0]
                            
                            # Create scene replacement video
                            final_video = self.advanced_processor.create_scene_replacement_video(
                                user_video_path, reference_video, final_music_file, user_instruction
                            )
                            
                            progress(0.6, desc="🎵 Adding audio and finalizing...")
                            
                            # Add audio
                            if final_music_file and os.path.exists(final_music_file):
                                from moviepy.editor import AudioFileClip
                                audio = AudioFileClip(final_music_file)
                                if audio.duration > final_video.duration:
                                    audio = audio.subclip(0, final_video.duration)
                                final_video = final_video.set_audio(audio)
                            
                            # Save the video
                            output_path = os.path.join("output", "scene_replacement_video.mp4")
                            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30, verbose=False, logger=None)
                            
                            output_paths = {"scene_replacement": output_path}
                            print(f"✅ Scene replacement video saved: {output_path}")
                            
                        except Exception as e:
                            print(f"Scene replacement failed, falling back to multiple versions: {e}")
                            # Fallback to multiple versions
                            versions = self.advanced_processor.generate_multiple_versions(
                                base_clips, final_music_file, user_instruction, target_duration, reference_video
                            )
                            
                            progress(0.6, desc="🎵 Syncing with music beats...")
                            
                            # Create final videos
                            output_paths = self.advanced_processor.create_final_video(
                                versions, final_music_file, "output"
                            )
                    else:
                        # Regular multiple versions mode
                        versions = self.advanced_processor.generate_multiple_versions(
                            base_clips, final_music_file, user_instruction, target_duration, reference_video
                        )
                        
                        progress(0.6, desc="🎵 Syncing with music beats...")
                        
                        # Create final videos
                        output_paths = self.advanced_processor.create_final_video(
                            versions, final_music_file, "output"
                        )
                    
                    # Handle scene replacement output
                    if "scene_replacement" in output_paths:
                        progress(1.0, desc="✅ Scene replacement video created!")
                        return [output_paths.get("scene_replacement"), None, None]
                    
                    progress(1.0, desc="✅ Multiple versions created!")
                    
                    # Return all versions
                    if "basic" in output_paths:
                        # Fallback case - return basic version in first slot
                        return [output_paths.get("basic"), None, None]
                    else:
                        # Normal case - return all three versions
                        return [
                            output_paths.get("energetic"),
                            output_paths.get("cinematic"), 
                            output_paths.get("natural")
                        ]
                    

                    print(f"Advanced processing failed, falling back to basic: {e}")
                    # Fallback to basic concatenation
                    from moviepy.editor import concatenate_videoclips
                    final_video = concatenate_videoclips(base_clips)
                    output_path = os.path.join("output", "nvidia_ai_video_basic.mp4")
                    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30, verbose=False, logger=None)
                    output_paths = {"basic": output_path}
                    
                    progress(1.0, desc="✅ Basic version created!")
                    return [output_paths.get("basic"), None, None]
                
                else:
                    # Use traditional single version processing
                    progress(0.1, desc="Processing with traditional method...")
                    
                    # Check if user provided intelligent instruction and NVIDIA is available
                    if user_instruction and user_instruction.strip() and self.nvidia_enabled:
                        # Use NVIDIA Cloud AI for intelligent editing
                        progress(0.3, desc="🌩️ NVIDIA: Understanding your instruction...")
                        
                        progress(0.6, desc="🎬 NVIDIA: Creating intelligent video...")
                        output_path = self.nvidia_engine.create_intelligent_video(
                            user_instruction=user_instruction,
                            media_files=media_files,
                            audio_path=final_music_file,
                            temp_dir=temp_dir
                        )
                        
                    elif user_instruction and user_instruction.strip() and not self.nvidia_enabled:
                        # NVIDIA not available but user wants AI
                        raise gr.Error("AI features require NVIDIA API key. Please set NVIDIA_API_KEY environment variable or use traditional mode (leave instruction empty).")
                        
                    else:
                        # Use traditional method
                        progress(0.1, desc="Processing with traditional method...")
                        
                        # Handle reference video (optional)
                        if reference_video and os.path.isfile(reference_video):
                            scene_data = self.video_analyzer.analyze_video(reference_video)
                        else:
                            # Create default scene data
                            scene_data = {"scenes": [{"duration": 3.0, "transition": "fade"}] * len(media_files)}
                        
                        progress(0.3, desc="Processing audio...")
                        beats = self.audio_processor.detect_beats(final_music_file)
                        
                        progress(0.5, desc="Preparing media files...")
                        output_path = self.video_generator.create_video(
                            scene_data,
                            beats,
                            media_files,
                            final_music_file,
                            temp_dir
                        )
                        
                        # Adjust to target duration if needed
                        if output_path and os.path.exists(output_path):
                            from moviepy.editor import VideoFileClip
                            video = VideoFileClip(output_path)
                            if video.duration > target_duration:
                                # Trim to target duration
                                trimmed_video = video.subclip(0, target_duration)
                                trimmed_path = output_path.replace('.mp4', '_trimmed.mp4')
                                trimmed_video.write_videofile(trimmed_path, codec="libx264", audio_codec="aac", fps=30, verbose=False, logger=None)
                                output_path = trimmed_path
                            elif video.duration < target_duration:
                                # Loop to fill duration
                                loops = int(np.ceil(target_duration / video.duration))
                                from moviepy.editor import concatenate_videoclips
                                extended_video = concatenate_videoclips([video] * loops).subclip(0, target_duration)
                                extended_path = output_path.replace('.mp4', '_extended.mp4')
                                extended_video.write_videofile(extended_path, codec="libx264", audio_codec="aac", fps=30, verbose=False, logger=None)
                                output_path = extended_path
                    
                    progress(1.0, desc="Done!")
                    return [output_path, None, None]
                
        except Exception as e:
            raise gr.Error(str(e))

def create_enhanced_ui():
    editor = EnhancedInstalreelsEditor()
    
    with gr.Blocks(title="Enhanced NVIDIA Cloud AI Video Editor") as interface:
        gr.Markdown("# 🚀 Enhanced NVIDIA Cloud AI Video Editor")
        gr.Markdown("Create professional Instagram Reels using a reference video as template with automatic lighting, intelligent cuts, and multiple style options!")
        
        with gr.Row():
            with gr.Column():
                # AI Instruction Section
                gr.Markdown("## 🧠 AI Video Instructions")
                ai_instruction = gr.Textbox(
                    label="🌩️ NVIDIA Cloud AI Instructions",
                    placeholder="e.g., 'Create an energetic gym video with fast cuts and beat sync' or 'Make a calm aesthetic slideshow with smooth fades'",
                    lines=3,
                    info="Leave empty to use traditional editing mode"
                )
                
                with gr.Accordion("Example Instructions", open=False):
                    gr.Markdown("""
                    **Energetic Style:** "Create an energetic video with fast cuts that sync to the music beats"
                    
                    **Cinematic Style:** "Make a cinematic video with dramatic lighting and smooth transitions"
                    
                    **Natural Style:** "Create a natural video with smooth fades and balanced lighting"
                    
                    **Vibrant Style:** "Make a vibrant video with bold colors and dynamic cuts"
                    
                    **Moody Style:** "Create a moody video with dramatic lighting and slow transitions"
                    """)
                
                gr.Markdown("## 📁 Media Files")
                reference_video = gr.File(
                    label="Reference Video (Required - This will be the template for your video)",
                    file_types=["video"],
                    type="filepath"
                )
                
                media_files = gr.File(
                    label="Your Photos/Videos",
                    file_count="multiple",
                    file_types=["image", "video"],
                    type="filepath"
                )
                
                # Music Options
                gr.Markdown("## 🎵 Music Options")
                music_choice = gr.Radio(
                    choices=["Use Reference Video Music", "Use My Own Music"],
                    value="Use Reference Video Music",
                    label="🎵 Choose Music Source",
                    info="Select which audio to use for your video"
                )
                
                music_file = gr.File(
                    label="Your Music File (Required if 'Use My Own Music' is selected)",
                    file_types=["audio"],
                    type="filepath",
                    visible=False
                )
                
                # Advanced Options
                gr.Markdown("## ⚙️ Advanced Options")
                generate_multiple_versions = gr.Checkbox(
                    label="🎬 Generate Multiple Versions",
                    value=True,
                    info="Create 3 different style versions for you to choose from"
                )
                
                target_duration = gr.Slider(
                    minimum=10.0,
                    maximum=60.0,
                    value=30.0,
                    step=5.0,
                    label="🎬 Target Video Duration (seconds)",
                    info="How long should the final video be?"
                )
                
                create_button = gr.Button("🚀 Create Enhanced Video", variant="primary", size="lg")
                
                # NVIDIA Status
                gr.Markdown("### 🌩️ NVIDIA Cloud Status")
                nvidia_status = gr.Textbox(
                    label="NVIDIA Build Status",
                    value="Set NVIDIA_API_KEY to enable cloud AI features",
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("## 🎬 Generated Videos")
                
                # Multiple output videos
                energetic_video = gr.Video(
                    label="⚡ Energetic Version (Fast-paced, Vibrant)",
                    visible=True
                )
                
                cinematic_video = gr.Video(
                    label="🎬 Cinematic Version (Dramatic, Professional)",
                    visible=True
                )
                
                natural_video = gr.Video(
                    label="🌿 Natural Version (Smooth, Balanced)",
                    visible=True
                )
                
                # Version descriptions
                with gr.Accordion("📋 Version Details", open=False):
                    gr.Markdown("""
                    **⚡ Energetic Version:**
                    - Fast-paced cuts (1.3x speed)
                    - Vibrant lighting and colors
                    - Perfect for fitness, dance, action content
                    
                    **🎬 Cinematic Version:**
                    - Dramatic lighting and contrast
                    - Subtle zoom effects
                    - Professional, movie-like quality
                    
                    **🌿 Natural Version:**
                    - Smooth transitions and fades
                    - Balanced, natural lighting
                    - Calm, aesthetic feel
                    """)
                
                # Advanced Features Info
                with gr.Accordion("🚀 Advanced Features", open=False):
                    gr.Markdown("""
                    **🎨 Automatic Lighting:**
                    - Cinematic, vibrant, moody, natural presets
                    - Automatic brightness, contrast, saturation adjustment
                    - Color temperature correction
                    
                    **🎵 Intelligent Beat Sync:**
                    - Automatic music beat detection
                    - Cuts synchronized to music rhythm
                    - Perfect timing for dynamic content
                    
                    **✂️ Smart Cuts:**
                    - AI-determined optimal clip durations
                    - Reference video timing integration
                    - Professional editing patterns
                    """)
        
        # Show/hide music file upload based on choice
        def toggle_music_upload(choice):
            return gr.File(visible=choice == "Use My Own Music")
        
        music_choice.change(
            fn=toggle_music_upload,
            inputs=[music_choice],
            outputs=[music_file]
        )
        
        create_button.click(
            fn=editor.process_video_with_options,
            inputs=[reference_video, media_files, music_file, ai_instruction, generate_multiple_versions, target_duration, music_choice],
            outputs=[energetic_video, cinematic_video, natural_video],
            show_progress=True
        )
    
    return interface

if __name__ == "__main__":
    # Ensure required directories exist
    for dir_path in ["input/reference", "input/media", "input/music", "output"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Launch the enhanced interface
    interface = create_enhanced_ui()
    interface.launch(share=False)
