"""
NVIDIA Image Generation Integration for Video Editor
Handles image generation with proper error handling and fallbacks
"""

import os
import requests
import json
import base64
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

class NVIDIAImageGenerator:
    """NVIDIA Build image generation with robust error handling."""
    
    def __init__(self, api_key: str = None):
        # Load config
        if os.path.exists("nvidia_config.env"):
            load_dotenv("nvidia_config.env")
        
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Available models (in order of preference)
        self.image_models = [
            "black-forest-labs/flux.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0", 
            "stabilityai/sdxl-turbo",
            "playgroundai/playground-v2.5-1024px-aesthetic"
        ]
        
        print("🎨 NVIDIA Image Generator initialized")

    def generate_video_thumbnail(self, video_description: str, style: str = "professional") -> Optional[str]:
        """
        Generate a thumbnail for the video based on description and style.
        """
        # Create style-specific prompt
        style_prompts = {
            "energetic": "dynamic, high-energy, vibrant colors, motion blur, fitness, gym",
            "aesthetic": "beautiful, artistic, soft lighting, pastel colors, minimalist, clean",
            "cinematic": "dramatic lighting, movie poster style, professional, cinematic",
            "professional": "clean, modern, business, corporate, professional lighting",
            "calm": "peaceful, serene, soft colors, relaxing, zen, minimalist"
        }
        
        base_prompt = f"Create a professional video thumbnail for: {video_description}"
        style_addition = style_prompts.get(style, "professional, high quality")
        
        full_prompt = f"{base_prompt}. Style: {style_addition}. High quality, 16:9 aspect ratio, no text overlay."
        
        return self._generate_image(full_prompt, width=1024, height=576)

    def generate_background_image(self, mood: str, color_palette: List[str] = None) -> Optional[str]:
        """
        Generate a background image based on mood and color palette.
        """
        color_desc = ""
        if color_palette:
            color_desc = f"using colors: {', '.join(color_palette)}"
        
        prompt = f"Abstract background with {mood} mood {color_desc}. Smooth gradients, no text, suitable for video overlay."
        
        return self._generate_image(prompt, width=1080, height=1920)

    def generate_text_overlay_background(self, text: str, style: str = "modern") -> Optional[str]:
        """
        Generate a background specifically designed for text overlays.
        """
        style_prompts = {
            "modern": "clean, minimalist, geometric shapes",
            "elegant": "sophisticated, luxury, gold accents",
            "energetic": "dynamic, bold, vibrant colors",
            "calm": "soft, peaceful, gentle gradients"
        }
        
        style_desc = style_prompts.get(style, "clean, modern")
        prompt = f"Text overlay background, {style_desc}, space for text '{text}', no existing text, professional design"
        
        return self._generate_image(prompt, width=1080, height=200)

    def _generate_image(self, prompt: str, width: int = 1024, height: int = 1024, steps: int = 4) -> Optional[str]:
        """
        Generate image using NVIDIA API with fallback models.
        """
        for model in self.image_models:
            try:
                print(f"🎨 Trying model: {model}")
                
                # Prepare payload
                payload = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "guidance_scale": 7.5
                }
                
                # Different endpoints for different models
                if "flux" in model:
                    url = f"https://ai.api.nvidia.com/v1/genai/{model}"
                else:
                    url = f"https://integrate.api.nvidia.com/v1/genai/{model}"
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if "images" in result:
                        image_data = result["images"][0]
                    elif "data" in result:
                        image_data = result["data"][0]["b64_json"]
                    else:
                        print(f"⚠️ Unexpected response format from {model}")
                        continue
                    
                    # Save image
                    output_path = self._save_generated_image(image_data, prompt)
                    if output_path:
                        print(f"✅ Image generated successfully with {model}")
                        return output_path
                
                elif response.status_code == 429:
                    print(f"⏳ Rate limited on {model}, waiting...")
                    time.sleep(5)
                    continue
                
                else:
                    print(f"❌ {model} failed: {response.status_code} - {response.text[:100]}")
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout with {model}, trying next...")
                continue
            except Exception as e:
                print(f"❌ Error with {model}: {str(e)[:100]}")
                continue
        
        print("❌ All image generation models failed")
        return None

    def _save_generated_image(self, image_data: str, prompt: str) -> Optional[str]:
        """
        Save generated image to file.
        """
        try:
            # Create output directory
            output_dir = Path("output/generated_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp = int(time.time())
            filename = f"generated_{safe_prompt}_{timestamp}.png"
            output_path = output_dir / filename
            
            # Decode and save
            image_bytes = base64.b64decode(image_data)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            return None

    def test_image_generation(self) -> bool:
        """
        Test image generation functionality.
        """
        print("🧪 Testing NVIDIA Image Generation...")
        
        test_prompts = [
            "Simple geometric background, blue and white gradient",
            "Abstract minimalist design, professional"
        ]
        
        for prompt in test_prompts:
            print(f"Testing: {prompt}")
            result = self._generate_image(prompt, width=512, height=512, steps=2)
            
            if result:
                print(f"✅ Test passed: {result}")
                return True
            else:
                print("❌ Test failed")
        
        return False

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available image generation models.
        """
        models_info = []
        
        for model in self.image_models:
            try:
                # Try to get model info
                url = f"https://integrate.api.nvidia.com/v1/models/{model}"
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    model_info = response.json()
                    models_info.append({
                        "id": model,
                        "status": "available",
                        "info": model_info
                    })
                else:
                    models_info.append({
                        "id": model,
                        "status": "unavailable",
                        "error": response.status_code
                    })
                    
            except Exception as e:
                models_info.append({
                    "id": model,
                    "status": "error",
                    "error": str(e)
                })
        
        return models_info

def test_nvidia_image_generation():
    """
    Test function for NVIDIA image generation.
    """
    print("🎨 NVIDIA Image Generation Test")
    print("=" * 40)
    
    try:
        generator = NVIDIAImageGenerator()
        
        # Test basic generation
        if generator.test_image_generation():
            print("✅ Basic image generation working!")
        else:
            print("❌ Basic image generation failed")
            return False
        
        # Test video thumbnail generation
        thumbnail = generator.generate_video_thumbnail(
            "energetic gym workout video", 
            "energetic"
        )
        
        if thumbnail:
            print(f"✅ Video thumbnail generated: {thumbnail}")
        else:
            print("❌ Video thumbnail generation failed")
        
        # Show available models
        models = generator.get_available_models()
        print(f"\n📋 Available models: {len(models)}")
        for model in models:
            status = model['status']
            print(f"   {model['id']}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        return False

if __name__ == "__main__":
    test_nvidia_image_generation()
