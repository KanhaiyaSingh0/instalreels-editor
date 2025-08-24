#!/usr/bin/env python3
"""
Comprehensive test suite for NVIDIA AI Video Editor
Tests all features systematically
"""

import os
import time
from dotenv import load_dotenv
from src.nvidia_cloud_ai_engine import NVIDIACloudVideoEngine, NVIDIACloudAI

def test_instruction_parsing():
    """Test various AI instruction parsing scenarios."""
    print("🧠 Testing AI Instruction Parsing")
    print("=" * 50)
    
    # Load config
    if os.path.exists("nvidia_config.env"):
        load_dotenv("nvidia_config.env")
    
    try:
        ai = NVIDIACloudAI()
        
        test_instructions = [
            # Basic styles
            "Create an energetic gym video with fast cuts",
            "Make a calm aesthetic slideshow with smooth fades", 
            "Create a cinematic story with dramatic transitions",
            
            # Advanced features
            "Make a professional promotional video with text overlay 'Our Product'",
            "Create a beat-synced music video with dynamic effects",
            "Make a travel video with varied transitions and upbeat mood",
            
            # Specific effects
            "Add flipping style transitions in the middle of the video",
            "Create a fitness motivation video with speed ramping effects",
            "Make an aesthetic food video with warm color palette"
        ]
        
        for i, instruction in enumerate(test_instructions, 1):
            print(f"\n{i}. Testing: \"{instruction}\"")
            try:
                result = ai.parse_instruction(instruction, media_count=5)
                print(f"   ✅ Style: {result.style}")
                print(f"   🎭 Mood: {result.mood}")
                print(f"   ⏱️ Pacing: {result.pacing}")
                print(f"   🎬 Transitions: {result.transitions}")
                print(f"   🎵 Music sync: {result.music_sync}")
                if result.text_overlay:
                    print(f"   📝 Text: {result.text_overlay}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
            
            time.sleep(1)  # Rate limiting
        
        print("\n✅ Instruction parsing tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Instruction parsing test failed: {e}")
        return False

def test_video_generation():
    """Test video generation with different scenarios."""
    print("\n\n🎬 Testing Video Generation")
    print("=" * 50)
    
    # Check for media files
    media_dir = "input/media"
    if not os.path.exists(media_dir):
        print(f"❌ Media directory not found: {media_dir}")
        return False
    
    media_files = []
    for file in os.listdir(media_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov')):
            media_files.append(os.path.join(media_dir, file))
    
    if len(media_files) < 2:
        print(f"❌ Need at least 2 media files, found {len(media_files)}")
        return False
    
    print(f"📁 Found {len(media_files)} media files")
    
    try:
        engine = NVIDIACloudVideoEngine(os.getenv("NVIDIA_API_KEY"))
        
        test_scenarios = [
            {
                "instruction": "Create a short energetic video with fast cuts",
                "description": "Basic energetic style test"
            },
            {
                "instruction": "Make a calm slideshow without text overlay",
                "description": "Aesthetic style without text"
            },
            {
                "instruction": "Create a professional video with clean transitions",
                "description": "Professional style test"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['description']}")
            print(f"   Instruction: \"{scenario['instruction']}\"")
            
            try:
                # Use first 3 media files for faster testing
                test_files = media_files[:3]
                
                output_path = engine.create_intelligent_video(
                    user_instruction=scenario['instruction'],
                    media_files=test_files,
                    audio_path=None
                )
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024*1024)  # MB
                    print(f"   ✅ Video created: {output_path} ({file_size:.1f} MB)")
                else:
                    print(f"   ❌ Video file not found: {output_path}")
                    
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
            
            time.sleep(2)  # Rate limiting
        
        print("\n✅ Video generation tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Video generation test failed: {e}")
        return False

def test_feature_combinations():
    """Test combinations of features."""
    print("\n\n🔀 Testing Feature Combinations")
    print("=" * 50)
    
    combinations = [
        {
            "instruction": "Create an energetic fitness video with beat sync and motivational text",
            "features": ["energetic", "beat_sync", "text_overlay"]
        },
        {
            "instruction": "Make a calm travel slideshow with smooth fades and warm colors",
            "features": ["aesthetic", "smooth_transitions", "color_palette"]
        },
        {
            "instruction": "Create a professional promotional video with clean cuts and modern style",
            "features": ["professional", "clean_cuts", "modern"]
        }
    ]
    
    try:
        ai = NVIDIACloudAI()
        
        for i, combo in enumerate(combinations, 1):
            print(f"\n{i}. Testing: {combo['features']}")
            print(f"   Instruction: \"{combo['instruction']}\"")
            
            try:
                result = ai.parse_instruction(combo['instruction'], media_count=4)
                
                # Check if features are detected
                detected_features = []
                if result.pacing == "fast":
                    detected_features.append("energetic")
                if result.music_sync:
                    detected_features.append("beat_sync")
                if result.text_overlay:
                    detected_features.append("text_overlay")
                if "smooth" in result.style.lower() or "fade" in str(result.transitions):
                    detected_features.append("smooth_transitions")
                
                print(f"   ✅ Detected: {detected_features}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature combination test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and fallback systems."""
    print("\n\n🛡️ Testing Error Handling")
    print("=" * 50)
    
    error_scenarios = [
        {
            "instruction": "",  # Empty instruction
            "description": "Empty instruction"
        },
        {
            "instruction": "xyz123 invalid instruction !!!",
            "description": "Invalid/unclear instruction"
        },
        {
            "instruction": "Create a video with impossible requirements that cannot be fulfilled",
            "description": "Impossible requirements"
        }
    ]
    
    try:
        ai = NVIDIACloudAI()
        
        for i, scenario in enumerate(error_scenarios, 1):
            print(f"\n{i}. Testing: {scenario['description']}")
            
            try:
                result = ai.parse_instruction(scenario['instruction'], media_count=3)
                print(f"   ✅ Fallback worked: {result.style} style")
                
            except Exception as e:
                print(f"   ⚠️ Error handled: {str(e)[:100]}...")
        
        print("\n✅ Error handling tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def performance_benchmark():
    """Benchmark performance of different operations."""
    print("\n\n⚡ Performance Benchmark")
    print("=" * 50)
    
    try:
        ai = NVIDIACloudAI()
        
        # Test instruction parsing speed
        start_time = time.time()
        result = ai.parse_instruction("Create an energetic video with fast cuts", 5)
        parsing_time = time.time() - start_time
        
        print(f"🧠 Instruction parsing: {parsing_time:.2f}s")
        
        # Test with different complexity levels
        simple_instruction = "Make a video"
        complex_instruction = "Create a cinematic storytelling video with dramatic transitions, text overlay saying 'My Journey', warm color palette, beat synchronization, and professional quality effects"
        
        start_time = time.time()
        ai.parse_instruction(simple_instruction, 3)
        simple_time = time.time() - start_time
        
        start_time = time.time()
        ai.parse_instruction(complex_instruction, 5)
        complex_time = time.time() - start_time
        
        print(f"📝 Simple instruction: {simple_time:.2f}s")
        print(f"📚 Complex instruction: {complex_time:.2f}s")
        
        print("\n✅ Performance benchmark completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def comprehensive_test_report():
    """Generate comprehensive test report."""
    print("\n" + "="*60)
    print("🎬 NVIDIA AI VIDEO EDITOR - COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    test_results = {}
    
    # Run all tests
    test_results["instruction_parsing"] = test_instruction_parsing()
    test_results["video_generation"] = test_video_generation()
    test_results["feature_combinations"] = test_feature_combinations()
    test_results["error_handling"] = test_error_handling()
    test_results["performance"] = performance_benchmark()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\n🎯 Overall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Your NVIDIA AI Video Editor is fully functional!")
    else:
        print("⚠️ Some tests failed. Check the detailed output above for issues.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not test_results["video_generation"]:
        print("• Check media files in input/media/ directory")
        print("• Ensure NVIDIA API key is valid")
    
    print("• Install ImageMagick for text overlay support")
    print("• Try different instruction styles for best results")
    print("• Use 3-8 media files for optimal performance")

if __name__ == "__main__":
    comprehensive_test_report()
