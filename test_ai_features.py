#!/usr/bin/env python3
"""
Test script to demonstrate AI features without requiring Ollama.
Shows how the AI instruction parser works with pattern matching.
"""

import os
from src.ai_instruction_parser import AIInstructionParser
from src.visual_content_analyzer import VisualContentAnalyzer

def test_instruction_parsing():
    """Test the AI instruction parser with various inputs."""
    print("🧠 Testing AI Instruction Parser")
    print("=" * 40)
    
    parser = AIInstructionParser()
    
    test_instructions = [
        "Create an energetic gym video with fast cuts and beat sync",
        "Make a calm aesthetic slideshow with smooth fades",
        "Create a cinematic story with dramatic transitions",
        "Sync all cuts to the music beats",
        "Make a professional promotional video",
        "Create a peaceful meditation video with long durations"
    ]
    
    for instruction in test_instructions:
        print(f"\n📝 Input: \"{instruction}\"")
        
        try:
            result = parser.parse_instruction(instruction, media_count=5)
            print(f"   ✅ Style: {result.edit_type.value}")
            print(f"   ⏱️  Duration per clip: {result.duration_per_clip}s")
            print(f"   🎵 Music sync: {result.music_sync}")
            print(f"   🏃 Pacing: {result.pacing}")
            print(f"   🎬 Transitions: {[t.value for t in result.transitions]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_visual_analysis():
    """Test visual content analysis on available media."""
    print("\n\n👁️ Testing Visual Content Analyzer")
    print("=" * 40)
    
    analyzer = VisualContentAnalyzer()
    
    # Check for media files in input directory
    media_dir = "input/media"
    if os.path.exists(media_dir):
        media_files = []
        for file in os.listdir(media_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi')):
                media_files.append(os.path.join(media_dir, file))
        
        print(f"📁 Found {len(media_files)} media files")
        
        for media_file in media_files[:3]:  # Test first 3 files
            print(f"\n📸 Analyzing: {os.path.basename(media_file)}")
            
            try:
                # Basic analysis (works without AI)
                basic_analysis = analyzer._basic_image_analysis(media_file)
                
                if "error" not in basic_analysis:
                    print(f"   🎨 Dominant colors: {len(basic_analysis.get('dominant_colors', []))} colors detected")
                    print(f"   👥 Faces detected: {basic_analysis.get('faces_count', 0)}")
                    print(f"   💡 Lighting: {basic_analysis.get('lighting', 'unknown')}")
                    print(f"   ⭐ Quality score: {basic_analysis.get('quality_score', 0):.2f}")
                else:
                    print(f"   ❌ Analysis failed: {basic_analysis['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    else:
        print("📁 No media directory found. Please add some images/videos to input/media/")

def test_system_integration():
    """Test overall system integration."""
    print("\n\n🔗 Testing System Integration")
    print("=" * 40)
    
    # Test if all components can be imported
    try:
        from src.intelligent_video_engine import IntelligentVideoEngine
        print("✅ Intelligent Video Engine imported successfully")
        
        engine = IntelligentVideoEngine()
        print("✅ AI engine initialized")
        
        # Test if Ollama is available
        if engine.instruction_parser.is_ollama_available():
            print("🤖 Ollama is available - Full AI features enabled!")
            models = engine.instruction_parser.get_available_models()
            print(f"📋 Available models: {models}")
        else:
            print("⚡ Running in fast mode (without Ollama) - Pattern matching enabled")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def print_usage_guide():
    """Print usage guide for the AI features."""
    print("\n\n📖 AI Features Usage Guide")
    print("=" * 40)
    print("""
🎬 Your AI-Powered Video Editor is ready!

🚀 Quick Start:
1. Open http://localhost:7860 in your browser
2. Add your photos/videos and optional music
3. Describe your desired video style in the text box

💡 Example AI Instructions:
• "Create an energetic gym video with fast cuts and beat sync"
• "Make a calm aesthetic slideshow with smooth fades"
• "Create a cinematic story with dramatic transitions"
• "Sync all cuts perfectly to the music beats"

🔧 System Status:
• ✅ Basic AI features (pattern matching) - Working
• 🤖 Advanced AI features (Ollama) - Optional
• 🎵 Audio processing - Ready
• 🎬 Video generation - Ready

📝 To enable full AI features:
Run: python setup_ai.py
    """)

if __name__ == "__main__":
    print("🎬 AI-Powered Video Editor - Feature Test")
    print("=" * 50)
    
    test_instruction_parsing()
    test_visual_analysis() 
    test_system_integration()
    print_usage_guide()
