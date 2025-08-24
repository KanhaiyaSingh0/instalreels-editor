#!/usr/bin/env python3
"""
Setup script for NVIDIA Cloud-Only AI Video Editor.
No local AI required - pure cloud processing!
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

def check_nvidia_api_key():
    """Check if NVIDIA API key is set."""
    # Load from config file
    if os.path.exists("nvidia_config.env"):
        load_dotenv("nvidia_config.env")
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key == "nvapi-your-key-here":
        return False, "NVIDIA API key not set in nvidia_config.env"
    
    # Test the API key
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models = response.json()
            return True, f"API key valid! {len(models.get('data', []))} models available"
        else:
            return False, f"API key invalid: {response.status_code} - {response.text[:100]}"
            
    except Exception as e:
        return False, f"API connection failed: {str(e)}"

def display_setup_guide():
    """Display setup guide for NVIDIA Build."""
    print("""
🌟 NVIDIA Cloud AI Video Editor Setup
=====================================

🚀 No Local AI Required - Pure Cloud Processing!

📋 Setup Steps:

1️⃣ Get NVIDIA Build Access:
   • Visit: https://build.nvidia.com
   • Sign up for free developer account
   • Navigate to "API" section
   • Create new API key

2️⃣ Configure API Key:
   
   Edit nvidia_config.env file:
   NVIDIA_API_KEY=nvapi-your-actual-key-here
   
   (Keep other settings as they are)

4️⃣ Verify Setup:
   python setup_nvidia_cloud.py

🌩️ Available NVIDIA AI Models:
   • Llama 3.1 70B: Advanced instruction understanding
   • Llama 3.2 Vision: Image/video analysis
   • Stable Diffusion XL: Image generation
   • CLIP: Vision understanding

✅ Benefits of NVIDIA Cloud:
   • No local GPU required
   • Enterprise-grade AI models
   • Always up-to-date models
   • Scalable cloud processing
   • No model downloads
   • Instant setup

💰 Pricing:
   • Free tier for development
   • Pay-per-use for production
   • Much cheaper than buying GPUs!
""")

def test_nvidia_features():
    """Test NVIDIA cloud features."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not set. Please follow setup guide above.")
        return False
    
    print("🧪 Testing NVIDIA Cloud AI Features...")
    
    try:
        from src.nvidia_cloud_ai_engine import NVIDIACloudAI
        
        # Initialize NVIDIA AI
        nvidia_ai = NVIDIACloudAI(api_key)
        
        print("✅ NVIDIA Cloud AI connection established!")
        
        # Test instruction parsing
        print("\n🧠 Testing Instruction Parsing...")
        instruction = nvidia_ai.parse_instruction(
            "Create an energetic workout video with fast cuts", 5
        )
        print(f"   ✅ Parsed: {instruction.style} style, {instruction.pacing} pacing")
        
        print("\n🌟 NVIDIA Cloud AI is ready for video editing!")
        return True
        
    except ImportError:
        print("❌ NVIDIA cloud engine not found. Make sure all files are in place.")
        return False
    except Exception as e:
        print(f"❌ NVIDIA test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🌟 NVIDIA Cloud AI Video Editor Setup")
    print("=" * 50)
    
    # Check if API key is set
    is_valid, message = check_nvidia_api_key()
    
    if is_valid:
        print(f"✅ {message}")
        
        # Test features
        if test_nvidia_features():
            print("\n🎉 Setup Complete! Your NVIDIA Cloud AI Video Editor is ready!")
            print("\n🚀 Next Steps:")
            print("1. Run: python app.py")
            print("2. Open: http://localhost:7860")
            print("3. Try AI instructions like:")
            print("   • 'Create an energetic gym video with fast cuts'")
            print("   • 'Make a calm aesthetic slideshow'")
            print("   • 'Create a cinematic story with dramatic transitions'")
        else:
            print("\n⚠️ Feature test failed. Check your setup.")
    else:
        print(f"❌ {message}")
        display_setup_guide()

if __name__ == "__main__":
    main()
