import time
from typing import Any, Dict, Optional


class EmotionService:
    def __init__(self):
        self.emotion_states = {
            "happy": {
                "intensity_range": (0.3, 1.0),
                "facial_expressions": ["smile", "laugh", "grin"],
                "body_language": ["relaxed", "open", "energetic"],
                "voice_characteristics": ["bright", "upbeat", "warm"],
            },
            "sad": {
                "intensity_range": (0.2, 0.8),
                "facial_expressions": ["frown", "downcast", "teary"],
                "body_language": ["slouched", "closed", "slow"],
                "voice_characteristics": ["soft", "low", "melancholic"],
            },
            "angry": {
                "intensity_range": (0.4, 1.0),
                "facial_expressions": ["scowl", "frown", "tense"],
                "body_language": ["rigid", "aggressive", "tense"],
                "voice_characteristics": ["sharp", "loud", "harsh"],
            },
            "surprised": {
                "intensity_range": (0.5, 1.0),
                "facial_expressions": ["wide_eyes", "raised_brows", "open_mouth"],
                "body_language": ["startled", "alert", "reactive"],
                "voice_characteristics": ["high_pitched", "sudden", "excited"],
            },
            "fearful": {
                "intensity_range": (0.3, 0.9),
                "facial_expressions": ["wide_eyes", "tense", "worried"],
                "body_language": ["cautious", "defensive", "trembling"],
                "voice_characteristics": ["trembling", "quiet", "nervous"],
            },
            "disgusted": {
                "intensity_range": (0.4, 0.8),
                "facial_expressions": ["wrinkled_nose", "scowl", "grimace"],
                "body_language": ["recoiling", "closed", "tense"],
                "voice_characteristics": ["nasal", "harsh", "disapproving"],
            },
            "neutral": {
                "intensity_range": (0.0, 0.3),
                "facial_expressions": ["relaxed", "natural", "calm"],
                "body_language": ["balanced", "neutral", "composed"],
                "voice_characteristics": ["even", "calm", "neutral"],
            },
        }

    def process_emotion(
        self, emotion: str, intensity: float, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an emotion and generate appropriate emotional state.
        """
        if emotion not in self.emotion_states:
            raise ValueError(f"Unknown emotion: {emotion}")

        emotion_config = self.emotion_states[emotion]
        min_intensity, max_intensity = emotion_config["intensity_range"]

        # Clamp intensity to valid range
        intensity = max(min_intensity, min(intensity, max_intensity))

        # Generate emotional state
        state = {
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": time.time(),
            "facial_expression": self._select_expression(
                emotion_config["facial_expressions"], intensity
            ),
            "body_language": self._select_body_language(
                emotion_config["body_language"], intensity
            ),
            "voice_characteristics": self._select_voice(
                emotion_config["voice_characteristics"], intensity
            ),
            "context": context or {},
        }

        # Add intensity-based modifiers
        state.update(self._get_intensity_modifiers(intensity))

        return state

    def _select_expression(self, expressions: list, intensity: float) -> str:
        """
        Select appropriate facial expression based on intensity.
        """
        if intensity < 0.3:
            return expressions[0]  # Subtle expression
        elif intensity < 0.7:
            return expressions[1]  # Moderate expression
        else:
            return expressions[2]  # Strong expression

    def _select_body_language(self, body_language: list, intensity: float) -> str:
        """
        Select appropriate body language based on intensity.
        """
        if intensity < 0.3:
            return body_language[0]  # Subtle body language
        elif intensity < 0.7:
            return body_language[1]  # Moderate body language
        else:
            return body_language[2]  # Strong body language

    def _select_voice(self, voice_characteristics: list, intensity: float) -> str:
        """
        Select appropriate voice characteristics based on intensity.
        """
        if intensity < 0.3:
            return voice_characteristics[0]  # Subtle voice
        elif intensity < 0.7:
            return voice_characteristics[1]  # Moderate voice
        else:
            return voice_characteristics[2]  # Strong voice

    def _get_intensity_modifiers(self, intensity: float) -> Dict[str, Any]:
        """
        Get additional modifiers based on emotion intensity.
        """
        return {
            "movement_speed": 1.0
            + (intensity - 0.5) * 0.5,  # Faster movements for higher intensity
            "gesture_frequency": intensity
            * 2.0,  # More frequent gestures for higher intensity
            "voice_volume": 0.5 + intensity * 0.5,  # Louder voice for higher intensity
            "reaction_time": 1.0
            - (intensity * 0.3),  # Faster reactions for higher intensity
        }
