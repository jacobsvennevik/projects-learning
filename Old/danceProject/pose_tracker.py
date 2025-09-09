from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import mediapipe as mp

class PoseTracker(ABC):
    """Abstract base class for pose tracking implementations."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the pose tracker and required resources."""
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process a single frame and return pose keypoints."""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release resources used by the pose tracker."""
        pass
    
    @abstractmethod
    def get_supported_keypoints(self) -> List[str]:
        """Return list of supported keypoint names."""
        pass

class BlazePoseTracker(PoseTracker):
    """Implementation using MediaPipe's BlazePose."""
    
    def __init__(self, model_complexity: int = 1):
        """
        Initialize BlazePose tracker.
        
        Args:
            model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
        """
        self.model_complexity = model_complexity
        self.mp_pose = mp.solutions.pose
        self.pose = None
        self.mp_drawing = mp.solutions.drawing_utils
        
    def initialize(self) -> bool:
        """Initialize the BlazePose model."""
        try:
            self.pose = self.mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return True
        except Exception as e:
            print(f"Failed to initialize BlazePose: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame and return pose keypoints.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing pose keypoints or None if detection fails
        """
        if self.pose is None:
            return None
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Extract keypoints
        keypoints = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[f"keypoint_{idx}"] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            }
            
        return keypoints
    
    def release(self) -> None:
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
            
    def get_supported_keypoints(self) -> List[str]:
        """Return list of supported keypoint names."""
        return [f"keypoint_{i}" for i in range(33)]  # BlazePose has 33 keypoints

class OpenPoseTracker(PoseTracker):
    """Placeholder for OpenPose implementation."""
    
    def __init__(self):
        """Initialize OpenPose tracker."""
        raise NotImplementedError("OpenPose integration not yet implemented")
    
    def initialize(self) -> bool:
        raise NotImplementedError
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        raise NotImplementedError
    
    def release(self) -> None:
        raise NotImplementedError
    
    def get_supported_keypoints(self) -> List[str]:
        raise NotImplementedError

def create_pose_tracker(tracker_type: str = "blazepose", **kwargs) -> PoseTracker:
    """
    Factory function to create a pose tracker instance.
    
    Args:
        tracker_type: Type of tracker to create ("blazepose" or "openpose")
        **kwargs: Additional arguments for tracker initialization
        
    Returns:
        PoseTracker instance
        
    Raises:
        ValueError: If tracker_type is not supported
    """
    if tracker_type.lower() == "blazepose":
        return BlazePoseTracker(**kwargs)
    elif tracker_type.lower() == "openpose":
        return OpenPoseTracker(**kwargs)
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}") 