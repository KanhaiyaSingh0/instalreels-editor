#!/usr/bin/env python3
"""
Analyze the generated output video to check if style transfer is working.
"""

import cv2
import numpy as np
import os

def analyze_video(video_path, video_name):
    """Analyze a video file and extract visual characteristics."""
    print(f"\n🔍 Analyzing {video_name}: {os.path.basename(video_path)}")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"📹 Video Properties:")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total frames: {frame_count}")
        
        # Analyze multiple frames
        frames_to_analyze = [0.1, 0.3, 0.5, 0.7, 0.9]  # 10%, 30%, 50%, 70%, 90%
        all_brightness = []
        all_saturation = []
        all_contrast = []
        all_colors = []
        
        for time_ratio in frames_to_analyze:
            frame_number = int(frame_count * time_ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Analyze brightness
                brightness = np.mean(frame_rgb)
                all_brightness.append(brightness)
                
                # Analyze color channels
                red_avg = np.mean(frame_rgb[:, :, 0])
                green_avg = np.mean(frame_rgb[:, :, 1])
                blue_avg = np.mean(frame_rgb[:, :, 2])
                all_colors.append([red_avg, green_avg, blue_avg])
                
                # Analyze saturation
                hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1])
                all_saturation.append(saturation)
                
                # Analyze contrast
                contrast = np.std(frame_rgb)
                all_contrast.append(contrast)
        
        cap.release()
        
        # Calculate averages
        avg_brightness = np.mean(all_brightness)
        avg_saturation = np.mean(all_saturation)
        avg_contrast = np.mean(all_contrast)
        avg_colors = np.mean(all_colors, axis=0)
        
        # Calculate warmth (red vs blue balance)
        warmth = avg_colors[0] / (avg_colors[2] + 1e-6)
        
        print(f"\n🎨 Visual Analysis Results:")
        print(f"   Brightness: {avg_brightness:.2f}")
        print(f"   Saturation: {avg_saturation:.2f}")
        print(f"   Contrast: {avg_contrast:.2f}")
        print(f"   Warmth: {warmth:.2f}")
        print(f"   Red channel: {avg_colors[0]:.2f}")
        print(f"   Green channel: {avg_colors[1]:.2f}")
        print(f"   Blue channel: {avg_colors[2]:.2f}")
        
        # Classify style
        if avg_brightness > 120 and avg_saturation > 100 and avg_contrast > 40:
            if warmth > 1.1:
                style = "energetic_fitness"
                description = "High-energy fitness video with vibrant, warm lighting"
            else:
                style = "vibrant_fitness"
                description = "Vibrant fitness video with dynamic lighting"
        elif avg_contrast > 50 and avg_brightness < 100:
            style = "cinematic"
            description = "Cinematic video with dramatic lighting"
        elif avg_brightness > 100 and avg_saturation < 80:
            style = "natural"
            description = "Natural, realistic video with balanced lighting"
        else:
            style = "energetic_fitness"
            description = "Energetic fitness video with dynamic lighting"
        
        print(f"\n🎯 Style Classification:")
        print(f"   Detected Style: {style}")
        print(f"   Description: {description}")
        
        return {
            "brightness": avg_brightness,
            "saturation": avg_saturation,
            "contrast": avg_contrast,
            "warmth": warmth,
            "colors": avg_colors,
            "style": style,
            "description": description
        }
        
    except Exception as e:
        print(f"❌ Error analyzing video: {str(e)}")
        return None

def compare_videos():
    """Compare reference video with generated output video."""
    print("🎬 Video Style Analysis & Comparison")
    print("=" * 70)
    
    # Analyze reference video
    reference_video = "input/reference/Look at this Trend 😎💪@fit_iffu_roxx AESTHETIC FITNESS ABS VEINS BODYBUILDING GYM WORKOUT MOTIV.mp4"
    ref_analysis = analyze_video(reference_video, "Reference Video")
    
    # Analyze generated output video
    output_video = "output/nvidia_ai_video_natural.mp4"
    out_analysis = analyze_video(output_video, "Generated Output Video")
    
    if ref_analysis and out_analysis:
        print(f"\n🔄 STYLE TRANSFER COMPARISON")
        print("=" * 50)
        
        # Compare key parameters
        brightness_diff = abs(ref_analysis["brightness"] - out_analysis["brightness"])
        saturation_diff = abs(ref_analysis["saturation"] - out_analysis["saturation"])
        contrast_diff = abs(ref_analysis["contrast"] - out_analysis["contrast"])
        warmth_diff = abs(ref_analysis["warmth"] - out_analysis["warmth"])
        
        print(f"📊 Parameter Differences (Reference vs Generated):")
        print(f"   Brightness: {brightness_diff:.2f} (Lower = Better match)")
        print(f"   Saturation: {saturation_diff:.2f} (Lower = Better match)")
        print(f"   Contrast: {contrast_diff:.2f} (Lower = Better match)")
        print(f"   Warmth: {warmth_diff:.2f} (Lower = Better match)")
        
        # Overall assessment
        total_diff = brightness_diff + saturation_diff + contrast_diff + warmth_diff
        if total_diff < 50:
            assessment = "EXCELLENT - Perfect style transfer! 🎉"
        elif total_diff < 100:
            assessment = "GOOD - Style transfer working well! ✅"
        elif total_diff < 150:
            assessment = "FAIR - Some style transfer happening ⚠️"
        else:
            assessment = "POOR - Style transfer not working ❌"
        
        print(f"\n🎯 Overall Assessment:")
        print(f"   {assessment}")
        print(f"   Total difference: {total_diff:.2f}")
        
        # Style match
        if ref_analysis["style"] == out_analysis["style"]:
            print(f"   Style Match: ✅ PERFECT - Both classified as {ref_analysis['style']}")
        else:
            print(f"   Style Match: ❌ MISMATCH - Ref: {ref_analysis['style']}, Output: {out_analysis['style']}")

if __name__ == "__main__":
    compare_videos()
