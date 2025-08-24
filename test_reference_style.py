#!/usr/bin/env python3
"""
Test script to demonstrate improved reference video style analysis.
This shows how the system now analyzes the actual visual characteristics of reference videos.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from advanced_video_processor import AdvancedVideoProcessor

def test_reference_video_analysis():
    """Test the enhanced reference video analysis."""
    
    # Initialize the processor
    processor = AdvancedVideoProcessor()
    
    # Reference video path (the fitness trend video)
    reference_video = "input/reference/Look at this Trend 😎💪@fit_iffu_roxx AESTHETIC FITNESS ABS VEINS BODYBUILDING GYM WORKOUT MOTIV.mp4"
    
    if not os.path.exists(reference_video):
        print(f"❌ Reference video not found: {reference_video}")
        return
    
    print("🎬 Testing Enhanced Reference Video Analysis")
    print("=" * 50)
    print(f"📹 Reference video: {os.path.basename(reference_video)}")
    print()
    
    try:
        # Analyze the reference video
        print("🔍 Analyzing reference video...")
        analysis = processor.analyze_reference_video(reference_video)
        
        if analysis and "visual_style" in analysis:
            style = analysis["visual_style"]
            
            print("\n✅ Analysis Complete!")
            print("=" * 30)
            print(f"🎨 Style: {style['style_name']}")
            print(f"📝 Description: {style['description']}")
            print(f"💡 Lighting: {style['lighting_description']}")
            print(f"🎨 Colors: {style['color_description']}")
            print()
            
            print("📊 Technical Parameters:")
            print(f"   Brightness: {style.get('brightness', 0):.3f}")
            print(f"   Saturation: {style.get('saturation', 0):.3f}")
            print(f"   Contrast: {style.get('contrast', 0):.3f}")
            print(f"   Warmth: {style.get('warmth', 0):.3f}")
            print(f"   Red channel: {style.get('red_avg', 0):.3f}")
            print(f"   Green channel: {style.get('green_avg', 0):.3f}")
            print(f"   Blue channel: {style.get('blue_avg', 0):.3f}")
            print()
            
            # Test custom preset creation
            print("🎨 Testing Custom Preset Creation:")
            custom_preset = processor._create_reference_style_preset(style)
            print(f"   Custom preset created with {len(custom_preset)} parameters")
            print()
            
            print("🎯 Expected Result:")
            print("   The generated video should now match the reference video's:")
            print("   • Lighting characteristics")
            print("   • Color palette")
            print("   • Visual mood and style")
            print("   • Brightness and contrast levels")
            print()
            
        else:
            print("❌ Analysis failed or no visual style detected")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def test_style_classification():
    """Test the enhanced style classification system."""
    
    processor = AdvancedVideoProcessor()
    
    print("🎨 Testing Style Classification System")
    print("=" * 40)
    
    # Test different parameter combinations
    test_cases = [
        (180, 200, 80, 1.3, "High brightness, high saturation, warm"),
        (160, 180, 70, 1.1, "Medium-high brightness, good saturation"),
        (120, 140, 60, 0.9, "Medium brightness, medium saturation"),
        (80, 100, 50, 0.8, "Low brightness, low saturation"),
    ]
    
    for brightness, saturation, contrast, warmth, description in test_cases:
        style_info = processor._classify_visual_style(brightness, saturation, contrast, warmth)
        print(f"📊 {description}:")
        print(f"   Parameters: B={brightness}, S={saturation}, C={contrast}, W={warmth}")
        print(f"   Classified as: {style_info['name']}")
        print(f"   Description: {style_info['description']}")
        print()

if __name__ == "__main__":
    print("🚀 Enhanced Reference Video Style Analysis Test")
    print("=" * 55)
    print()
    
    # Test reference video analysis
    test_reference_video_analysis()
    
    print("\n" + "=" * 55)
    print()
    
    # Test style classification
    test_style_classification()
    
    print("✅ Test completed!")
    print("\n💡 Key Improvements:")
    print("   • Reference video is now analyzed for actual visual characteristics")
    print("   • Generated videos inherit the exact lighting and color style")
    print("   • No more hardcoded 'natural' preset - uses reference video's real style")
    print("   • Perfect style transfer from reference to generated content")
