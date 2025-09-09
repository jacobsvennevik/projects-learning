import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class AngleFeatureExtractor:
    """Extracts angle-based features from joint coordinates."""
    
    def __init__(self):
        """Initialize the angle feature extractor."""
        # Define joint pairs for angle calculation
        self.angle_pairs = [
            # Arms
            ('left_shoulder', 'left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow', 'right_wrist'),
            
            # Legs
            ('left_hip', 'left_knee', 'left_ankle'),
            ('right_hip', 'right_knee', 'right_ankle'),
            
            # Torso
            ('left_shoulder', 'left_hip', 'left_knee'),
            ('right_shoulder', 'right_hip', 'right_knee'),
            
            # Head
            ('nose', 'left_eye', 'left_ear'),
            ('nose', 'right_eye', 'right_ear')
        ]
        
        # Define joint indices mapping
        self.joint_indices = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
    
    def _compute_angle(self, 
                      p1: np.ndarray, 
                      p2: np.ndarray, 
                      p3: np.ndarray) -> float:
        """
        Compute the angle between three points.
        
        Args:
            p1: First point (vertex)
            p2: Second point (middle)
            p3: Third point (end)
            
        Returns:
            Angle in radians
        """
        # Handle zero vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Compute angle
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cos_angle)
        
        # Handle case where vectors are perpendicular
        if np.isclose(cos_angle, 0.0):
            return np.pi/2
            
        return angle
    
    def extract_features(self, 
                        joints: np.ndarray,
                        n_workers: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract angle features from joint coordinates.
        
        Args:
            joints: Array of shape (n_frames, n_joints, 3) containing joint coordinates
            n_workers: Optional number of parallel workers (not used in this implementation)
            
        Returns:
            Dictionary containing:
                - 'angles': Array of shape (n_frames, n_angles) containing 2D angles
                - 'angles_3d': Array of shape (n_frames, n_angles, 3) containing 3D angles
        """
        if joints.size == 0:
            raise ValueError("Empty input array")
            
        if joints.shape[1] != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} joints, got {joints.shape[1]}")
            
        n_frames = joints.shape[0]
        n_angles = len(self.angle_pairs)
        
        # Initialize output arrays
        angles = np.zeros((n_frames, n_angles))
        angles_3d = np.zeros((n_frames, n_angles, 3))
        
        # Compute angles for each frame
        for frame in range(n_frames):
            for i, (joint1, joint2, joint3) in enumerate(self.angle_pairs):
                # Get joint indices
                idx1 = self.joint_indices[joint1]
                idx2 = self.joint_indices[joint2]
                idx3 = self.joint_indices[joint3]
                
                # Get joint coordinates
                p1 = joints[frame, idx1]
                p2 = joints[frame, idx2]
                p3 = joints[frame, idx3]
                
                # Handle NaN values
                if np.any(np.isnan([p1, p2, p3])):
                    angles[frame, i] = 0.0
                    angles_3d[frame, i] = [0.0, 0.0, 0.0]
                    continue
                
                # Compute angle
                angle = self._compute_angle(p1, p2, p3)
                angles[frame, i] = angle
                
                # For 3D angles, use the same angle for all dimensions
                angles_3d[frame, i] = [angle, angle, angle]
        
        return {
            'angles': angles,
            'angles_3d': angles_3d
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of the extracted features.
        
        Returns:
            List of feature names
        """
        return [f"angle_{j2}" for _, j2, _ in self.angle_pairs] 