from typing import Dict, List
import numpy as np
from .base import FeatureExtractor

class DistanceFeatureExtractor(FeatureExtractor):
    """Extracts relative joint distances from pose keypoints."""
    
    def __init__(self):
        """Initialize the distance feature extractor."""
        super().__init__()
        self.feature_names = [
            "left_arm_length",
            "right_arm_length",
            "left_leg_length",
            "right_leg_length",
            "shoulder_width",
            "hip_width",
            "torso_length"
        ]
        self.feature_dimensions = {name: 1 for name in self.feature_names}
        
        # Define joint pairs for distance computation
        self.joint_pairs = {
            "left_arm_length": (5, 9),     # Left shoulder to wrist
            "right_arm_length": (6, 10),    # Right shoulder to wrist
            "left_leg_length": (13, 17),    # Left hip to ankle
            "right_leg_length": (14, 18),   # Right hip to ankle
            "shoulder_width": (5, 6),       # Between shoulders
            "hip_width": (11, 12),          # Between hips
            "torso_length": (0, 11)         # Neck to mid-hip
        }
    
    def extract(self, joints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract relative joint distances from pose keypoints.
        
        Args:
            joints: Array of shape (num_joints, 3) containing joint positions
            
        Returns:
            Dictionary mapping distance names to their values
        """
        if not self._validate_joints(joints):
            return {name: np.array([0.0]) for name in self.feature_names}
            
        distances = {}
        for name, (i, j) in self.joint_pairs.items():
            if i < len(joints) and j < len(joints):
                distance = self._compute_distance(joints[i], joints[j])
                distances[name] = np.array([distance])
            else:
                distances[name] = np.array([0.0])
                
        return distances 