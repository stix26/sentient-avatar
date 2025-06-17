from typing import Dict, Any, Optional
import time
import random
import math

class PhysicalService:
    def __init__(self):
        self.physical_actions = {
            "walking": {
                "duration_range": (1.0, 10.0),
                "movement_patterns": ["natural", "purposeful", "casual"],
                "body_posture": ["upright", "balanced", "relaxed"],
                "gait_characteristics": ["smooth", "rhythmic", "steady"]
            },
            "running": {
                "duration_range": (0.5, 5.0),
                "movement_patterns": ["dynamic", "energetic", "powerful"],
                "body_posture": ["forward_leaning", "athletic", "focused"],
                "gait_characteristics": ["quick", "bouncy", "efficient"]
            },
            "jumping": {
                "duration_range": (0.2, 1.0),
                "movement_patterns": ["explosive", "controlled", "graceful"],
                "body_posture": ["compressed", "extended", "balanced"],
                "gait_characteristics": ["springy", "powerful", "coordinated"]
            },
            "dancing": {
                "duration_range": (2.0, 15.0),
                "movement_patterns": ["fluid", "rhythmic", "expressive"],
                "body_posture": ["dynamic", "artistic", "flowing"],
                "gait_characteristics": ["graceful", "musical", "coordinated"]
            },
            "gesturing": {
                "duration_range": (0.5, 3.0),
                "movement_patterns": ["expressive", "meaningful", "natural"],
                "body_posture": ["engaged", "open", "communicative"],
                "gait_characteristics": ["articulate", "fluid", "purposeful"]
            },
            "idle": {
                "duration_range": (1.0, 30.0),
                "movement_patterns": ["subtle", "natural", "relaxed"],
                "body_posture": ["comfortable", "balanced", "natural"],
                "gait_characteristics": ["minimal", "occasional", "casual"]
            }
        }

        self.muscle_groups = {
            "upper_body": ["shoulders", "arms", "chest", "back"],
            "core": ["abdomen", "lower_back", "hips"],
            "lower_body": ["thighs", "calves", "feet"]
        }

    def process_physical(
        self,
        action: str,
        parameters: Dict[str, Any],
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a physical action and generate appropriate physical state.
        """
        if action not in self.physical_actions:
            raise ValueError(f"Unknown physical action: {action}")

        physical_config = self.physical_actions[action]
        min_duration, max_duration = physical_config["duration_range"]
        
        # Use provided duration or generate random one
        if duration is None:
            duration = random.uniform(min_duration, max_duration)
        else:
            duration = max(min_duration, min(duration, max_duration))

        # Generate physical state
        state = {
            "action": action,
            "timestamp": time.time(),
            "duration": duration,
            "movement_pattern": self._select_movement(physical_config["movement_patterns"]),
            "body_posture": self._select_posture(physical_config["body_posture"]),
            "gait_characteristics": self._select_gait(physical_config["gait_characteristics"]),
            "parameters": parameters
        }

        # Add physical characteristics
        state.update(self._get_physical_characteristics(action, duration, parameters))

        return state

    def _select_movement(self, movement_patterns: list) -> str:
        """
        Select appropriate movement pattern.
        """
        return random.choice(movement_patterns)

    def _select_posture(self, body_postures: list) -> str:
        """
        Select appropriate body posture.
        """
        return random.choice(body_postures)

    def _select_gait(self, gait_characteristics: list) -> str:
        """
        Select appropriate gait characteristics.
        """
        return random.choice(gait_characteristics)

    def _get_physical_characteristics(
        self,
        action: str,
        duration: float,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get additional physical characteristics based on action and parameters.
        """
        # Calculate energy expenditure
        energy_expenditure = self._calculate_energy_expenditure(action, duration)
        
        # Generate muscle tension
        muscle_tension = self._generate_muscle_tension(action)
        
        # Calculate movement quality
        movement_quality = self._calculate_movement_quality(action, parameters)

        return {
            "energy_expenditure": energy_expenditure,
            "muscle_tension": muscle_tension,
            "movement_quality": movement_quality,
            "breathing_rate": self._calculate_breathing_rate(action, energy_expenditure),
            "balance": self._calculate_balance(action, movement_quality),
            "coordination": self._calculate_coordination(action, movement_quality)
        }

    def _calculate_energy_expenditure(self, action: str, duration: float) -> float:
        """
        Calculate energy expenditure based on action and duration.
        """
        energy_rates = {
            "walking": 0.5,
            "running": 1.5,
            "jumping": 2.0,
            "dancing": 1.2,
            "gesturing": 0.3,
            "idle": 0.1
        }
        return energy_rates.get(action, 0.5) * duration

    def _generate_muscle_tension(self, action: str) -> Dict[str, float]:
        """
        Generate muscle tension for different muscle groups.
        """
        base_tensions = {
            "walking": {"upper_body": 0.3, "core": 0.4, "lower_body": 0.6},
            "running": {"upper_body": 0.5, "core": 0.6, "lower_body": 0.8},
            "jumping": {"upper_body": 0.4, "core": 0.7, "lower_body": 0.9},
            "dancing": {"upper_body": 0.6, "core": 0.5, "lower_body": 0.7},
            "gesturing": {"upper_body": 0.7, "core": 0.3, "lower_body": 0.2},
            "idle": {"upper_body": 0.2, "core": 0.2, "lower_body": 0.2}
        }

        tensions = base_tensions.get(action, {"upper_body": 0.3, "core": 0.3, "lower_body": 0.3})
        return {group: tension + random.uniform(-0.1, 0.1) for group, tension in tensions.items()}

    def _calculate_movement_quality(
        self,
        action: str,
        parameters: Dict[str, Any]
    ) -> float:
        """
        Calculate movement quality based on action and parameters.
        """
        base_quality = {
            "walking": 0.8,
            "running": 0.7,
            "jumping": 0.6,
            "dancing": 0.9,
            "gesturing": 0.8,
            "idle": 0.9
        }.get(action, 0.7)

        # Adjust quality based on parameters
        if "speed" in parameters:
            base_quality *= (1.0 - abs(0.5 - parameters["speed"]))
        if "precision" in parameters:
            base_quality *= parameters["precision"]

        return min(1.0, max(0.0, base_quality))

    def _calculate_breathing_rate(self, action: str, energy_expenditure: float) -> float:
        """
        Calculate breathing rate based on action and energy expenditure.
        """
        base_rates = {
            "walking": 12,
            "running": 20,
            "jumping": 25,
            "dancing": 18,
            "gesturing": 14,
            "idle": 10
        }
        return base_rates.get(action, 12) * (1.0 + energy_expenditure * 0.2)

    def _calculate_balance(self, action: str, movement_quality: float) -> float:
        """
        Calculate balance based on action and movement quality.
        """
        base_balance = {
            "walking": 0.8,
            "running": 0.7,
            "jumping": 0.6,
            "dancing": 0.9,
            "gesturing": 0.8,
            "idle": 0.9
        }.get(action, 0.7)
        return base_balance * movement_quality

    def _calculate_coordination(self, action: str, movement_quality: float) -> float:
        """
        Calculate coordination based on action and movement quality.
        """
        base_coordination = {
            "walking": 0.8,
            "running": 0.7,
            "jumping": 0.6,
            "dancing": 0.9,
            "gesturing": 0.8,
            "idle": 0.9
        }.get(action, 0.7)
        return base_coordination * movement_quality 