from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class FeatureExtractor(ABC):
    """Abstract base class for pose feature extraction."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names: List[str] = []
        self.feature_dimensions: Dict[str, int] = {}
    
    @abstractmethod
    def extract(self, joints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from joint positions.
        
        Args:
            joints: Array of shape (num_joints, 3) containing joint positions in 3D space
            
        Returns:
            Dictionary mapping feature names to their corresponding values
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features this extractor produces.
        
        Returns:
            List of feature names
        """
        return self.feature_names
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of each feature.
        
        Returns:
            Dictionary mapping feature names to their dimensions
        """
        return self.feature_dimensions
    
    def _validate_joints(self, joints: np.ndarray) -> bool:
        """
        Validate the input joint positions.
        
        Args:
            joints: Array of joint positions
            
        Returns:
            True if joints are valid, False otherwise
        """
        if not isinstance(joints, np.ndarray):
            return False
        if len(joints.shape) != 2 or joints.shape[1] != 3:
            return False
        return True
    
    def _compute_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two 3D points.
        
        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)
            
        Returns:
            Euclidean distance between the points
        """
        return np.linalg.norm(p1 - p2)
    
    def _compute_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle between three 3D points in degrees.
        
        Args:
            p1: First point (x, y, z)
            p2: Middle point (x, y, z)
            p3: Third point (x, y, z)
            
        Returns:
            Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
            
        # Compute angle using dot product
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle) 