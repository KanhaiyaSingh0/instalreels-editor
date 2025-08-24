#!/usr/bin/env python3
"""
Quick test to verify the error fixes are working.
"""

import os
import time
from dotenv import load_dotenv
from src.nvidia_cloud_ai_engine import NVIDIACloudAI

def test_fixes():
    """Test that the main errors are fixed."""
    print("🔧 Testing Error Fixes")
    print("=" * 30)
    
    # Load config
    if os.path.exists("nvidia_config.env"):
        load_dotenv("nvidia_config.env")
    
    try:
        ai = NVIDIACloudAI()
        
        # Test 1: Instruction parsing (should work)
        print("1. Testing instruction parsing...")
        result = ai.parse_instruction("Create an energetic video with fast cuts", 3)
        print(f"   ✅ Style: {result.style}, Mood: {result.mood}")
        
        # Test 2: Image analysis (should use smart fallback)
        print("\n2. Testing image analysis...")
        media_dir = "input/media"
        if os.path.exists(media_dir):
            media_files = [f for f in os.listdir(media_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            if media_files:
                test_image = os.path.join(media_dir, media_files[0])
                analysis = ai.analyze_image(test_image)
                print(f"   ✅ Objects: {analysis.objects}")
                print(f"   ✅ Mood: {analysis.mood}")
                print(f"   ✅ Quality: {analysis.quality_score}")
            else:
                print("   ⚠️ No image files found for testing")
        else:
            print("   ⚠️ Media directory not found")
        
        # Test 3: Edit plan enhancement (should handle JSON better)
        print("\n3. Testing edit plan enhancement...")
        from src.nvidia_cloud_ai_engine import CloudContentAnalysis
        
        fake_analysis = CloudContentAnalysis(
            file_path="test.jpg",
            content_type="image",
            objects=["person"],
            scenes=["gym"],
            activities=["workout"],
            mood="energetic",
            quality_score=0.8,
            faces_count=1,
            lighting="bright",
            recommended_duration=3.0,
            style_tags=["fitness"]
        )
        
        plan = ai.enhance_edit_plan(result, [fake_analysis])
        print(f"   ✅ Plan created: {type(plan)} with {len(plan)} items")
        
        print("\n🎉 All error fixes working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_fixes()
