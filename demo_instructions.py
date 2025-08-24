#!/usr/bin/env python3
"""
Demo script showcasing different AI instruction examples for the video editor.
"""

demo_instructions = {
    "energetic_gym": {
        "instruction": "Create an energetic gym workout video with fast cuts that sync to the music beats. Use dynamic transitions and keep the energy high throughout.",
        "description": "Perfect for fitness content, workout videos, sports highlights",
        "style": "Fast-paced, beat-synchronized, high energy"
    },
    
    "aesthetic_lifestyle": {
        "instruction": "Make a calm aesthetic video with smooth fades and longer transitions. Focus on the visual beauty and create a peaceful, Instagram-worthy slideshow.",
        "description": "Great for lifestyle, beauty, travel, and aesthetic content",
        "style": "Smooth, elegant, visually pleasing"
    },
    
    "cinematic_story": {
        "instruction": "Create a cinematic storytelling video with dramatic transitions and text overlay 'My Journey'. Use slow motion effects and build emotional connection.",
        "description": "Perfect for personal stories, documentaries, emotional content",
        "style": "Dramatic, story-driven, emotionally engaging"
    },
    
    "promotional_business": {
        "instruction": "Make a professional promotional video with clean cuts and modern transitions. Keep it business-appropriate with consistent pacing.",
        "description": "Ideal for business content, product showcases, professional presentations",
        "style": "Clean, professional, consistent"
    },
    
    "beat_sync_music": {
        "instruction": "Sync all cuts perfectly to the music beats for maximum impact. Create a rhythmic video where every transition matches the audio.",
        "description": "Best for music videos, dance content, rhythm-focused videos",
        "style": "Rhythm-driven, perfectly synchronized"
    },
    
    "calm_meditation": {
        "instruction": "Create a very slow and calm video with gentle fades. Make it peaceful and meditative with longer clip durations.",
        "description": "Perfect for meditation, relaxation, mindfulness content",
        "style": "Peaceful, slow, meditative"
    },
    
    "travel_adventure": {
        "instruction": "Make an adventurous travel video with varied transitions including slides and zooms. Show the excitement of exploration with medium-paced cuts.",
        "description": "Great for travel vlogs, adventure content, exploration videos",
        "style": "Adventurous, varied, engaging"
    },
    
    "food_aesthetic": {
        "instruction": "Create a food video with appetizing close-ups and smooth transitions. Make it look delicious and Instagram-worthy with warm aesthetic.",
        "description": "Perfect for food content, cooking videos, restaurant promotions",
        "style": "Appetizing, warm, aesthetic"
    }
}

def print_demo_instructions():
    """Print all demo instructions in a formatted way."""
    print("🎬 AI-Powered Video Editor - Demo Instructions")
    print("=" * 60)
    print()
    
    for key, demo in demo_instructions.items():
        print(f"📝 {key.replace('_', ' ').title()}")
        print(f"   Instruction: \"{demo['instruction']}\"")
        print(f"   Best for: {demo['description']}")
        print(f"   Style: {demo['style']}")
        print()

def get_instruction_by_type(instruction_type: str) -> str:
    """Get instruction text by type."""
    return demo_instructions.get(instruction_type, {}).get("instruction", "")

if __name__ == "__main__":
    print_demo_instructions()
    
    print("💡 Usage Tips:")
    print("- Be specific about the mood and style you want")
    print("- Mention if you want text overlays")
    print("- Specify pacing (fast, slow, medium)")
    print("- Indicate if you want beat synchronization")
    print("- Describe the target audience or use case")
    print()
    print("🚀 Try these instructions in your AI-powered video editor!")
