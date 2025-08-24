"""
AI-powered instruction parser that understands natural language video editing commands.
Uses local LLM (Ollama) for processing user instructions.
"""

import json
import re
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class EditType(Enum):
    BASIC_SLIDESHOW = "basic_slideshow"
    DYNAMIC_CUTS = "dynamic_cuts"
    BEAT_SYNC = "beat_sync"
    STORYTELLING = "storytelling"
    AESTHETIC = "aesthetic"
    ENERGETIC = "energetic"
    CALM = "calm"
    PROMOTIONAL = "promotional"

class TransitionType(Enum):
    CUT = "cut"
    FADE = "fade"
    SLIDE = "slide"
    ZOOM = "zoom"
    SPIN = "spin"
    DISSOLVE = "dissolve"

@dataclass
class EditInstruction:
    edit_type: EditType
    style: str
    duration_per_clip: float
    transitions: List[TransitionType]
    effects: List[str]
    pacing: str  # "slow", "medium", "fast"
    mood: str
    special_requirements: List[str]
    music_sync: bool
    text_overlay: Optional[str] = None
    brand_colors: Optional[List[str]] = None

class AIInstructionParser:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.model_name = "llama3.1"  # You can change to llama3.2-vision for VLM
        
        # Predefined patterns for quick parsing
        self.style_patterns = {
            "cinematic": EditType.STORYTELLING,
            "dynamic": EditType.DYNAMIC_CUTS,
            "energetic": EditType.ENERGETIC,
            "calm": EditType.CALM,
            "beat": EditType.BEAT_SYNC,
            "music": EditType.BEAT_SYNC,
            "slideshow": EditType.BASIC_SLIDESHOW,
            "promotional": EditType.PROMOTIONAL,
            "aesthetic": EditType.AESTHETIC
        }
        
        self.transition_patterns = {
            "fade": TransitionType.FADE,
            "cut": TransitionType.CUT,
            "slide": TransitionType.SLIDE,
            "zoom": TransitionType.ZOOM,
            "spin": TransitionType.SPIN,
            "dissolve": TransitionType.DISSOLVE
        }

    def parse_instruction(self, user_input: str, media_count: int = 0) -> EditInstruction:
        """
        Parse natural language instruction into structured editing parameters.
        """
        try:
            # First, try quick pattern matching
            quick_result = self._quick_parse(user_input, media_count)
            if quick_result:
                return quick_result
            
            # If quick parsing fails, use LLM
            llm_result = self._llm_parse(user_input, media_count)
            return llm_result
            
        except Exception as e:
            print(f"Error parsing instruction: {e}")
            # Fallback to default
            return self._get_default_instruction(media_count)

    def _quick_parse(self, user_input: str, media_count: int) -> Optional[EditInstruction]:
        """
        Quick pattern-based parsing for common instructions.
        """
        user_input_lower = user_input.lower()
        
        # Detect edit type
        edit_type = EditType.BASIC_SLIDESHOW
        for pattern, etype in self.style_patterns.items():
            if pattern in user_input_lower:
                edit_type = etype
                break
        
        # Detect transitions
        transitions = []
        for pattern, ttype in self.transition_patterns.items():
            if pattern in user_input_lower:
                transitions.append(ttype)
        
        if not transitions:
            transitions = [TransitionType.FADE]  # Default
        
        # Detect pacing
        pacing = "medium"
        if any(word in user_input_lower for word in ["fast", "quick", "rapid", "energetic"]):
            pacing = "fast"
        elif any(word in user_input_lower for word in ["slow", "calm", "relaxed", "peaceful"]):
            pacing = "slow"
        
        # Detect duration preferences
        duration_per_clip = 3.0  # Default
        if "long" in user_input_lower or "slow" in user_input_lower:
            duration_per_clip = 4.0
        elif "short" in user_input_lower or "fast" in user_input_lower:
            duration_per_clip = 2.0
        
        # Detect music sync
        music_sync = any(word in user_input_lower for word in ["beat", "music", "sync", "rhythm"])
        
        # Extract text overlay
        text_overlay = None
        text_match = re.search(r'"([^"]*)"', user_input)
        if text_match:
            text_overlay = text_match.group(1)
        
        return EditInstruction(
            edit_type=edit_type,
            style=edit_type.value,
            duration_per_clip=duration_per_clip,
            transitions=transitions,
            effects=[],
            pacing=pacing,
            mood=pacing,
            special_requirements=[],
            music_sync=music_sync,
            text_overlay=text_overlay
        )

    def _llm_parse(self, user_input: str, media_count: int) -> EditInstruction:
        """
        Use LLM to parse complex instructions.
        """
        prompt = f"""
        You are an AI video editor assistant. Parse the following user instruction for creating a video reel:

        User Input: "{user_input}"
        Number of media files: {media_count}

        Respond with a JSON object containing these fields:
        {{
            "edit_type": "one of: basic_slideshow, dynamic_cuts, beat_sync, storytelling, aesthetic, energetic, calm, promotional",
            "style": "brief description of visual style",
            "duration_per_clip": "number (seconds per media item, typically 2-5)",
            "transitions": ["list of transitions: cut, fade, slide, zoom, spin, dissolve"],
            "effects": ["list of effects requested"],
            "pacing": "one of: slow, medium, fast",
            "mood": "mood description",
            "special_requirements": ["any specific requirements"],
            "music_sync": "boolean - true if should sync to music beats",
            "text_overlay": "text to add to video if any, null otherwise"
        }}

        Examples:
        - "Make an energetic gym workout video with fast cuts" -> {{"edit_type": "energetic", "pacing": "fast", "transitions": ["cut"], "music_sync": true}}
        - "Create a calm slideshow with fades" -> {{"edit_type": "calm", "pacing": "slow", "transitions": ["fade"], "music_sync": false}}
        
        Respond only with valid JSON.
        """
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                
                # Parse JSON response
                parsed_data = json.loads(llm_output)
                
                return EditInstruction(
                    edit_type=EditType(parsed_data.get("edit_type", "basic_slideshow")),
                    style=parsed_data.get("style", "standard"),
                    duration_per_clip=float(parsed_data.get("duration_per_clip", 3.0)),
                    transitions=[TransitionType(t) for t in parsed_data.get("transitions", ["fade"])],
                    effects=parsed_data.get("effects", []),
                    pacing=parsed_data.get("pacing", "medium"),
                    mood=parsed_data.get("mood", "neutral"),
                    special_requirements=parsed_data.get("special_requirements", []),
                    music_sync=parsed_data.get("music_sync", False),
                    text_overlay=parsed_data.get("text_overlay")
                )
            
        except Exception as e:
            print(f"LLM parsing failed: {e}")
        
        # Fallback to quick parse
        return self._quick_parse(user_input, media_count) or self._get_default_instruction(media_count)

    def _get_default_instruction(self, media_count: int) -> EditInstruction:
        """
        Get default editing instruction.
        """
        return EditInstruction(
            edit_type=EditType.BASIC_SLIDESHOW,
            style="standard slideshow",
            duration_per_clip=3.0,
            transitions=[TransitionType.FADE],
            effects=[],
            pacing="medium",
            mood="neutral",
            special_requirements=[],
            music_sync=False
        )

    def suggest_improvements(self, user_input: str, current_result: Dict) -> List[str]:
        """
        Suggest improvements based on the current result.
        """
        suggestions = []
        
        if not current_result.get("music_sync") and any(word in user_input.lower() for word in ["music", "beat"]):
            suggestions.append("Consider enabling music synchronization for better rhythm")
        
        if len(current_result.get("transitions", [])) == 1:
            suggestions.append("Try mixing different transitions for more visual interest")
        
        if current_result.get("duration_per_clip", 3) > 4:
            suggestions.append("Consider shorter clips for better engagement on social media")
        
        return suggestions

    def is_ollama_available(self) -> bool:
        """
        Check if Ollama is running and accessible.
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models.
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
