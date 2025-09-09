from typing import Dict, List
import numpy as np
from .base import FeatureExtractor

class PlaneFeatureExtractor(FeatureExtractor):
    """Extracts joint projection planes from pose keypoints."""
    
    def __init__(self):
        """Initialize the plane feature extractor."""
        super().__init__()
        self.feature_names = [
            "frontal_plane_angle",    # Angle between body and frontal plane
            "sagittal_plane_angle",   # Angle between body and sagittal plane
            "transverse_plane_angle"  # Angle between body and transverse plane
        ]
        self.feature_dimensions = {name: 1 for name in self.feature_names}
        
        # Define joint triplets for plane computation
        self.plane_triplets = {
            "frontal_plane_angle": (0, 11, 12),    # Neck and hips for frontal plane
            "sagittal_plane_angle": (5, 6, 11),    # Shoulders and left hip for sagittal
            "transverse_plane_angle": (0, 5, 6)    # Neck and shoulders for transverse
        }
    
    def _compute_plane_normal(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """
        Compute the normal vector of a plane defined by three points.
        
        Args:
            p1, p2, p3: Three points defining the plane
            
        Returns:
            Normal vector of the plane
        """
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        return normal
    
    def _compute_plane_angle(self, normal1: np.ndarray, normal2: np.ndarray) -> float:
        """
        Compute the angle between two plane normals.
        
        Args:
            normal1, normal2: Normal vectors of two planes
            
        Returns:
            Angle between planes in degrees
        """
        cos_angle = np.abs(np.dot(normal1, normal2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def extract(self, joints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract joint projection planes from pose keypoints.
        
        Args:
            joints: Array of shape (num_joints, 3) containing joint positions
            
        Returns:
            Dictionary mapping plane names to their angles
        """
        if not self._validate_joints(joints):
            return {name: np.array([0.0]) for name in self.feature_names}
            
        planes = {}
        
        # Reference planes (frontal, sagittal, transverse)
        reference_normals = {
            "frontal_plane_angle": np.array([1.0, 0.0, 0.0]),    # X-axis
            "sagittal_plane_angle": np.array([0.0, 1.0, 0.0]),   # Y-axis
            "transverse_plane_angle": np.array([0.0, 0.0, 1.0])  # Z-axis
        }
        
        for name, (i, j, k) in self.plane_triplets.items():
            if i < len(joints) and j < len(joints) and k < len(joints):
                # Compute body plane normal
                body_normal = self._compute_plane_normal(joints[i], joints[j], joints[k])
                # Compute angle with reference plane
                angle = self._compute_plane_angle(body_normal, reference_normals[name])
                planes[name] = np.array([angle])
            else:
                planes[name] = np.array([0.0])
                
        return planes 