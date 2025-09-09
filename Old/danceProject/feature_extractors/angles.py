from typing import Dict, List
import numpy as np
from .base import FeatureExtractor

class AngleFeatureExtractor(FeatureExtractor):
    """Extracts joint angles from pose keypoints."""
    
    def __init__(self):
        """Initialize the angle feature extractor."""
        super().__init__()
        self.feature_names = [
            "angle_left_elbow",
            "angle_right_elbow",
            "angle_left_shoulder",
            "angle_right_shoulder",
            "angle_left_hip",
            "angle_right_hip",
            "angle_left_knee",
            "angle_right_knee"
        ]
        self.feature_dimensions = {name: 1 for name in self.feature_names}
        
        # Define joint triplets for angle computation using MediaPipe indices
        self.angle_pairs = {
            "angle_left_elbow": (11, 13, 15),    # Left shoulder, elbow, wrist
            "angle_right_elbow": (12, 14, 16),   # Right shoulder, elbow, wrist
            "angle_left_shoulder": (23, 11, 13), # Left hip, shoulder, elbow
            "angle_right_shoulder": (24, 12, 14), # Right hip, shoulder, elbow
            "angle_left_hip": (23, 25, 27),      # Left hip, knee, ankle
            "angle_right_hip": (24, 26, 28),     # Right hip, knee, ankle
            "angle_left_knee": (25, 27, 29),     # Left knee, ankle, heel
            "angle_right_knee": (26, 28, 30)     # Right knee, ankle, heel
        }
    
    def _compute_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle between three points in 3D space.
        
        Args:
            p1: First point
            p2: Middle point (vertex)
            p3: Third point
            
        Returns:
            Angle in radians
        """
        # Compute vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Compute angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cos_angle)
        
        return angle
    
    def extract(self, joints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract joint angles from pose keypoints.
        
        Args:
            joints: Array of shape (num_joints, 3) containing joint positions
            
        Returns:
            Dictionary mapping angle names to their values
        """
        if not self._validate_joints(joints):
            return {name: np.array([0.0]) for name in self.feature_names}
            
        angles = {}
        for name, (i, j, k) in self.angle_pairs.items():
            if i < len(joints) and j < len(joints) and k < len(joints):
                angle = self._compute_angle(joints[i], joints[j], joints[k])
                angles[name] = np.array([angle])
            else:
                angles[name] = np.array([0.0])
                
        return angles 