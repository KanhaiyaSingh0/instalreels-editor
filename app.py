import gradio as gr
from src.video_analyzer import VideoAnalyzer
from src.audio_processor import AudioProcessor
from src.video_generator import VideoGenerator
from src.nvidia_cloud_ai_engine import NVIDIACloudVideoEngine, check_nvidia_setup
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

class InstalreelsEditor:
    def __init__(self):
        self.video_analyzer = VideoAnalyzer()
        self.audio_processor = AudioProcessor()
        self.video_generator = VideoGenerator()
        
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
        
    def process_video(
        self,
        reference_video,
        media_files,
        music_file=None,
        user_instruction="",
        progress=gr.Progress()
    ):
        """
        Process video with optional AI-powered intelligent editing.
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
                raise gr.Error("Please upload a valid reference video file.")
            media_files = [m for m in (media_files or []) if isinstance(m, str) and os.path.isfile(m)]
            if not media_files:
                raise gr.Error("Please upload at least one photo or video in 'Your Photos/Videos'.")
            if music_file is not None and not os.path.isfile(music_file):
                raise gr.Error("Uploaded music file path is invalid.")

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Check if user provided intelligent instruction and NVIDIA is available
                if user_instruction and user_instruction.strip() and self.nvidia_enabled:
                    # Use NVIDIA Cloud AI for intelligent editing
                    progress(0.1, desc="🌩️ NVIDIA: Understanding your instruction...")
                    
                    progress(0.3, desc="👁️ NVIDIA: Analyzing your media content...")
                    
                    progress(0.6, desc="🎬 NVIDIA: Creating intelligent video...")
                    output_path = self.nvidia_engine.create_intelligent_video(
                        user_instruction=user_instruction,
                        media_files=media_files,
                        audio_path=music_file,
                        temp_dir=temp_dir
                    )
                    
                elif user_instruction and user_instruction.strip() and not self.nvidia_enabled:
                    # NVIDIA not available but user wants AI
                    raise gr.Error("AI features require NVIDIA API key. Please set NVIDIA_API_KEY environment variable or use traditional mode (leave instruction empty).")
                    
                else:
                    # Use traditional method
                    progress(0.1, desc="Analyzing reference video...")
                    # Analyze reference video
                    scene_data = self.video_analyzer.analyze_video(reference_video)
                    
                    progress(0.3, desc="Processing audio...")
                    # Process audio
                    if music_file:
                        beats = self.audio_processor.detect_beats(music_file)
                    else:
                        # Extract audio from reference video if no music provided
                        beats = self.audio_processor.extract_and_analyze_audio(reference_video)
                    
                    progress(0.5, desc="Preparing media files...")
                    # Generate output video
                    output_path = self.video_generator.create_video(
                        scene_data,
                        beats,
                        media_files,
                        music_file if music_file else reference_video,
                        temp_dir
                    )
                
                progress(1.0, desc="Done!")
                return output_path
                
        except Exception as e:
            # Surface a user-friendly error in the UI without returning invalid file paths
            raise gr.Error(str(e))

def create_ui():
    editor = InstalreelsEditor()
    
    with gr.Blocks(title="NVIDIA Cloud AI Video Editor") as interface:
        gr.Markdown("# 🌟 NVIDIA Cloud AI Video Editor")
        gr.Markdown("Create Instagram-style videos with NVIDIA's powerful cloud AI! No local AI setup required - pure cloud processing power!")
        
        with gr.Row():
            with gr.Column():
                # AI Instruction Section
                gr.Markdown("## 🧠 AI Video Instructions")
                ai_instruction = gr.Textbox(
                    label="🌩️ NVIDIA Cloud AI Instructions (Requires NVIDIA API Key)",
                    placeholder="e.g., 'Create an energetic gym video with fast cuts and beat sync' or 'Make a calm aesthetic slideshow with smooth fades'",
                    lines=3,
                    info="Leave empty to use traditional editing mode"
                )
                
                with gr.Accordion("Example Instructions", open=False):
                    gr.Markdown("""
                    **Energetic Style:** "Create an energetic video with fast cuts that sync to the music beats"
                    
                    **Aesthetic Style:** "Make a calm aesthetic video with smooth fades and longer transitions"
                    
                    **Storytelling:** "Create a cinematic storytelling video with dramatic transitions"
                    
                    **Promotional:** "Make a promotional video with text overlay and professional cuts"
                    
                    **Beat Sync:** "Sync all cuts to the music beats for a dynamic effect"
                    """)
                
                gr.Markdown("## 📁 Media Files")
                reference_video = gr.File(
                    label="Reference Video (Optional for AI mode - Used for timing reference in traditional mode)",
                    file_types=["video"],
                    type="filepath"
                )
                
                media_files = gr.File(
                    label="Your Photos/Videos",
                    file_count="multiple",
                    file_types=["image", "video"],
                    type="filepath"
                )
                
                music_file = gr.File(
                    label="Music File (Optional)",
                    file_types=["audio"],
                    type="filepath"
                )
                
                create_button = gr.Button("🌟 Create Video with NVIDIA AI", variant="primary", size="lg")
                
                # NVIDIA Status
                gr.Markdown("### 🌩️ NVIDIA Cloud Status")
                nvidia_status = gr.Textbox(
                    label="NVIDIA Build Status",
                    value="Set NVIDIA_API_KEY to enable cloud AI features",
                    interactive=False
                )
            
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                
                # NVIDIA Insights
                with gr.Accordion("🌩️ NVIDIA Analysis", open=False):
                    nvidia_insights = gr.Textbox(
                        label="NVIDIA Cloud Analysis",
                        placeholder="NVIDIA AI insights about your media will appear here...",
                        lines=5,
                        interactive=False
                    )
                
        create_button.click(
            fn=editor.process_video,
            inputs=[reference_video, media_files, music_file, ai_instruction],
            outputs=output_video
        )
    
    return interface

if __name__ == "__main__":
    # Ensure required directories exist
    for dir_path in ["input/reference", "input/media", "input/music", "output"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Launch the interface
    interface = create_ui()
    interface.launch(share=True)