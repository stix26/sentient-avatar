import asyncio
import time
from typing import Any, AsyncGenerator, Dict

from src.models.avatar import Avatar


class StreamingService:
    def __init__(self):
        self.update_interval = 0.1  # 100ms between updates
        self.max_duration = 3600  # 1 hour maximum streaming duration

    async def stream_avatar_updates(
        self, avatar: Avatar
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time updates for an avatar.
        """
        start_time = time.time()
        last_update = start_time

        while (time.time() - start_time) < self.max_duration:
            current_time = time.time()
            time_since_last_update = current_time - last_update

            if time_since_last_update >= self.update_interval:
                # Generate update
                update = self._generate_update(avatar, current_time - start_time)
                yield update
                last_update = current_time

            # Small delay to prevent CPU overuse
            await asyncio.sleep(0.01)

    def _generate_update(self, avatar: Avatar, elapsed_time: float) -> Dict[str, Any]:
        """
        Generate a real-time update for the avatar.
        """
        # Base update with avatar state
        update = {
            "timestamp": time.time(),
            "elapsed_time": elapsed_time,
            "avatar_id": avatar.id,
            "name": avatar.name,
        }

        # Add current states if available
        if avatar.current_emotion:
            update["emotion"] = self._process_emotion_state(avatar.current_emotion)
        if avatar.current_cognitive_state:
            update["cognitive"] = self._process_cognitive_state(
                avatar.current_cognitive_state
            )
        if avatar.current_physical_state:
            update["physical"] = self._process_physical_state(
                avatar.current_physical_state
            )

        # Add micro-expressions and subtle movements
        update["micro_expressions"] = self._generate_micro_expressions(avatar)
        update["subtle_movements"] = self._generate_subtle_movements(avatar)

        return update

    def _process_emotion_state(self, emotion_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and enhance emotion state for streaming.
        """
        return {
            "emotion": emotion_state["emotion"],
            "intensity": emotion_state["intensity"],
            "facial_expression": emotion_state["facial_expression"],
            "body_language": emotion_state["body_language"],
            "voice_characteristics": emotion_state["voice_characteristics"],
            "movement_speed": emotion_state.get("movement_speed", 1.0),
            "gesture_frequency": emotion_state.get("gesture_frequency", 1.0),
            "voice_volume": emotion_state.get("voice_volume", 1.0),
            "reaction_time": emotion_state.get("reaction_time", 1.0),
        }

    def _process_cognitive_state(
        self, cognitive_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and enhance cognitive state for streaming.
        """
        return {
            "operation": cognitive_state["operation"],
            "duration": cognitive_state["duration"],
            "facial_expression": cognitive_state["facial_expression"],
            "body_language": cognitive_state["body_language"],
            "voice_characteristics": cognitive_state["voice_characteristics"],
            "focus_level": cognitive_state.get("focus_level", 0.7),
            "processing_speed": cognitive_state.get("processing_speed", 1.0),
            "attention_span": cognitive_state.get("attention_span", 1.0),
        }

    def _process_physical_state(self, physical_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and enhance physical state for streaming.
        """
        return {
            "action": physical_state["action"],
            "duration": physical_state["duration"],
            "movement_pattern": physical_state["movement_pattern"],
            "body_posture": physical_state["body_posture"],
            "gait_characteristics": physical_state["gait_characteristics"],
            "energy_expenditure": physical_state.get("energy_expenditure", 0.0),
            "muscle_tension": physical_state.get("muscle_tension", {}),
            "movement_quality": physical_state.get("movement_quality", 1.0),
            "breathing_rate": physical_state.get("breathing_rate", 12.0),
            "balance": physical_state.get("balance", 1.0),
            "coordination": physical_state.get("coordination", 1.0),
        }

    def _generate_micro_expressions(self, avatar: Avatar) -> Dict[str, Any]:
        """
        Generate subtle micro-expressions based on current states.
        """
        micro_expressions = {
            "eye_movements": [],
            "facial_muscles": [],
            "breathing_pattern": "normal",
        }

        # Add eye movements
        if avatar.current_emotion:
            emotion = avatar.current_emotion["emotion"]
            if emotion in ["happy", "excited"]:
                micro_expressions["eye_movements"].extend(["bright", "sparkling"])
            elif emotion in ["sad", "tired"]:
                micro_expressions["eye_movements"].extend(["soft", "drooping"])
            elif emotion in ["angry", "focused"]:
                micro_expressions["eye_movements"].extend(["intense", "focused"])

        # Add facial muscle movements
        if avatar.current_cognitive_state:
            operation = avatar.current_cognitive_state["operation"]
            if operation in ["thinking", "analyzing"]:
                micro_expressions["facial_muscles"].extend(
                    ["subtle_frown", "brow_furrow"]
                )
            elif operation in ["learning", "remembering"]:
                micro_expressions["facial_muscles"].extend(
                    ["slight_smile", "raised_brows"]
                )

        return micro_expressions

    def _generate_subtle_movements(self, avatar: Avatar) -> Dict[str, Any]:
        """
        Generate subtle body movements based on current states.
        """
        subtle_movements = {
            "posture_adjustments": [],
            "weight_shifts": [],
            "breathing_movements": "normal",
        }

        # Add posture adjustments
        if avatar.current_physical_state:
            action = avatar.current_physical_state["action"]
            if action in ["walking", "running"]:
                subtle_movements["posture_adjustments"].extend(
                    ["slight_lean", "rhythmic_sway"]
                )
            elif action in ["idle", "thinking"]:
                subtle_movements["posture_adjustments"].extend(
                    ["micro_shifts", "comfort_adjustments"]
                )

        # Add weight shifts
        if avatar.current_emotion:
            emotion = avatar.current_emotion["emotion"]
            if emotion in ["nervous", "anxious"]:
                subtle_movements["weight_shifts"].extend(
                    ["slight_rocking", "weight_transfer"]
                )
            elif emotion in ["confident", "excited"]:
                subtle_movements["weight_shifts"].extend(
                    ["forward_lean", "energetic_shift"]
                )

        return subtle_movements
