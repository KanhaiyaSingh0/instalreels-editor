#!/usr/bin/env python3
"""
Setup script for AI-powered video editor.
Installs and configures Ollama with required models.
"""

import subprocess
import sys
import platform
import requests
import time
import os

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama based on the operating system."""
    system = platform.system().lower()
    
    print("🤖 Installing Ollama...")
    
    if system == "linux" or system == "darwin":  # Linux or macOS
        try:
            subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], check=True)
            subprocess.run(['sh'], input="curl -fsSL https://ollama.ai/install.sh | sh", shell=True, text=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to install Ollama automatically.")
            print("Please install Ollama manually from: https://ollama.ai/")
            return False
    
    elif system == "windows":
        print("🪟 For Windows, please download and install Ollama from:")
        print("https://ollama.ai/download/windows")
        print("After installation, restart this script.")
        return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False
    
    return True

def wait_for_ollama():
    """Wait for Ollama service to be ready."""
    print("⏳ Waiting for Ollama service to start...")
    
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✅ Ollama service is ready!")
                return True
        except requests.RequestException:
            pass
        
        time.sleep(1)
        print(f"⏳ Still waiting... ({i+1}/30)")
    
    print("❌ Ollama service didn't start in time. Please check the installation.")
    return False

def pull_model(model_name):
    """Pull a specific Ollama model."""
    print(f"📥 Downloading {model_name} model...")
    
    try:
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ {model_name} downloaded successfully!")
            return True
        else:
            print(f"❌ Failed to download {model_name}: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout downloading {model_name}. The model might be large.")
        return False
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def setup_models():
    """Setup required AI models."""
    models_to_install = [
        "llama3.1",  # For instruction parsing
        "llava",     # For visual analysis (optional but recommended)
    ]
    
    print("🧠 Setting up AI models...")
    
    # Check which models are already available
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        installed_models = result.stdout.lower()
    except:
        installed_models = ""
    
    for model in models_to_install:
        if model.lower() in installed_models:
            print(f"✅ {model} is already installed")
        else:
            success = pull_model(model)
            if not success and model == "llava":
                print(f"⚠️ {model} is optional. You can continue without it.")
            elif not success:
                print(f"❌ Failed to install required model: {model}")
                return False
    
    return True

def test_ai_functionality():
    """Test if AI functionality is working."""
    print("🧪 Testing AI functionality...")
    
    try:
        # Test basic LLM
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",
                "prompt": "Say 'AI test successful'",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "test successful" in result.get("response", "").lower():
                print("✅ Basic AI functionality is working!")
                return True
        
        print("❌ AI test failed")
        return False
        
    except Exception as e:
        print(f"❌ AI test error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎬 AI-Powered Video Editor Setup")
    print("=" * 40)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("📦 Ollama not found. Installing...")
        if not install_ollama():
            sys.exit(1)
    else:
        print("✅ Ollama is already installed")
    
    # Start Ollama service (if not running)
    try:
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Give it time to start
    except:
        pass
    
    # Wait for service to be ready
    if not wait_for_ollama():
        print("\n💡 Try running 'ollama serve' in another terminal, then run this script again.")
        sys.exit(1)
    
    # Setup models
    if not setup_models():
        print("❌ Failed to setup AI models")
        sys.exit(1)
    
    # Test functionality
    if not test_ai_functionality():
        print("❌ AI functionality test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Make sure Ollama service is running: ollama serve")
    print("2. Run your video editor: python app.py")
    print("3. Try AI instructions like:")
    print("   - 'Create an energetic video with fast cuts'")
    print("   - 'Make a calm aesthetic slideshow'")
    print("   - 'Sync all cuts to the music beats'")

if __name__ == "__main__":
    main()
