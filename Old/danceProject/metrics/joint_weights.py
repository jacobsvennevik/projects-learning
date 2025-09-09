from typing import Dict

# MediaPipe BlazePose joint indices
JOINT_INDICES = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}

# Ballet weights - emphasis on leg movements and posture
BALLET_WEIGHTS = {
    # Core and posture
    JOINT_INDICES['nose']: 1.2,
    JOINT_INDICES['left_shoulder']: 1.0,
    JOINT_INDICES['right_shoulder']: 1.0,
    JOINT_INDICES['left_hip']: 2.0,
    JOINT_INDICES['right_hip']: 2.0,
    
    # Legs - high importance for ballet
    JOINT_INDICES['left_knee']: 2.5,
    JOINT_INDICES['right_knee']: 2.5,
    JOINT_INDICES['left_ankle']: 2.0,
    JOINT_INDICES['right_ankle']: 2.0,
    JOINT_INDICES['left_heel']: 1.5,
    JOINT_INDICES['right_heel']: 1.5,
    
    # Arms - moderate importance
    JOINT_INDICES['left_elbow']: 1.0,
    JOINT_INDICES['right_elbow']: 1.0,
    JOINT_INDICES['left_wrist']: 0.8,
    JOINT_INDICES['right_wrist']: 0.8,
}

# Hip-hop weights - emphasis on upper body and rhythm
HIPHOP_WEIGHTS = {
    # Core and rhythm
    JOINT_INDICES['left_hip']: 1.0,
    JOINT_INDICES['right_hip']: 1.0,
    JOINT_INDICES['left_shoulder']: 2.0,
    JOINT_INDICES['right_shoulder']: 2.0,
    
    # Arms - high importance for hip-hop
    JOINT_INDICES['left_elbow']: 2.5,
    JOINT_INDICES['right_elbow']: 2.5,
    JOINT_INDICES['left_wrist']: 2.0,
    JOINT_INDICES['right_wrist']: 2.0,
    
    # Head and neck
    JOINT_INDICES['nose']: 1.2,
    
    # Legs - moderate importance
    JOINT_INDICES['left_knee']: 0.8,
    JOINT_INDICES['right_knee']: 0.8,
    JOINT_INDICES['left_ankle']: 0.6,
    JOINT_INDICES['right_ankle']: 0.6,
}

# Ballroom weights - balanced emphasis on all movements
BALLROOM_WEIGHTS = {
    # Core and posture
    JOINT_INDICES['nose']: 1.3,
    JOINT_INDICES['left_shoulder']: 1.4,
    JOINT_INDICES['right_shoulder']: 1.4,
    JOINT_INDICES['left_hip']: 1.4,
    JOINT_INDICES['right_hip']: 1.4,
    
    # Arms - important for leading/following
    JOINT_INDICES['left_elbow']: 1.3,
    JOINT_INDICES['right_elbow']: 1.3,
    JOINT_INDICES['left_wrist']: 1.2,
    JOINT_INDICES['right_wrist']: 1.2,
    
    # Legs - important for steps and rhythm
    JOINT_INDICES['left_knee']: 1.3,
    JOINT_INDICES['right_knee']: 1.3,
    JOINT_INDICES['left_ankle']: 1.2,
    JOINT_INDICES['right_ankle']: 1.2,
}

# Contemporary weights - emphasis on fluid movements
CONTEMPORARY_WEIGHTS = {
    # Core and spine
    JOINT_INDICES['nose']: 1.3,
    JOINT_INDICES['left_shoulder']: 1.4,
    JOINT_INDICES['right_shoulder']: 1.4,
    JOINT_INDICES['left_hip']: 1.4,
    JOINT_INDICES['right_hip']: 1.4,
    
    # Arms - important for expression
    JOINT_INDICES['left_elbow']: 1.3,
    JOINT_INDICES['right_elbow']: 1.3,
    JOINT_INDICES['left_wrist']: 1.2,
    JOINT_INDICES['right_wrist']: 1.2,
    
    # Legs - important for movement quality
    JOINT_INDICES['left_knee']: 1.3,
    JOINT_INDICES['right_knee']: 1.3,
    JOINT_INDICES['left_ankle']: 1.2,
    JOINT_INDICES['right_ankle']: 1.2,
    
    # Feet - important for connection to ground
    JOINT_INDICES['left_heel']: 1.1,
    JOINT_INDICES['right_heel']: 1.1,
}

# Latin weights - emphasis on hip movements and rhythm
LATIN_WEIGHTS = {
    # Core and hips - high importance for Latin dance
    JOINT_INDICES['left_hip']: 1.5,
    JOINT_INDICES['right_hip']: 1.5,
    JOINT_INDICES['left_shoulder']: 1.4,
    JOINT_INDICES['right_shoulder']: 1.4,
    
    # Arms - important for styling
    JOINT_INDICES['left_elbow']: 1.3,
    JOINT_INDICES['right_elbow']: 1.3,
    JOINT_INDICES['left_wrist']: 1.2,
    JOINT_INDICES['right_wrist']: 1.2,
    
    # Legs - important for rhythm
    JOINT_INDICES['left_knee']: 1.3,
    JOINT_INDICES['right_knee']: 1.3,
    JOINT_INDICES['left_ankle']: 1.2,
    JOINT_INDICES['right_ankle']: 1.2,
}

# Tap weights - emphasis on foot movements
TAP_WEIGHTS = {
    # Feet - highest importance for tap
    JOINT_INDICES['left_ankle']: 1.5,
    JOINT_INDICES['right_ankle']: 1.5,
    JOINT_INDICES['left_heel']: 1.4,
    JOINT_INDICES['right_heel']: 1.4,
    JOINT_INDICES['left_foot_index']: 1.3,
    JOINT_INDICES['right_foot_index']: 1.3,
    
    # Legs - important for rhythm
    JOINT_INDICES['left_knee']: 1.3,
    JOINT_INDICES['right_knee']: 1.3,
    
    # Core - moderate importance
    JOINT_INDICES['left_hip']: 1.2,
    JOINT_INDICES['right_hip']: 1.2,
    JOINT_INDICES['left_shoulder']: 1.1,
    JOINT_INDICES['right_shoulder']: 1.1,
}

# Jazz weights - emphasis on dynamic movements
JAZZ_WEIGHTS = {
    # Core and posture
    JOINT_INDICES['nose']: 1.2,
    JOINT_INDICES['left_shoulder']: 1.4,
    JOINT_INDICES['right_shoulder']: 1.4,
    JOINT_INDICES['left_hip']: 1.4,
    JOINT_INDICES['right_hip']: 1.4,
    
    # Arms - important for style
    JOINT_INDICES['left_elbow']: 1.3,
    JOINT_INDICES['right_elbow']: 1.3,
    JOINT_INDICES['left_wrist']: 1.2,
    JOINT_INDICES['right_wrist']: 1.2,
    
    # Legs - important for technique
    JOINT_INDICES['left_knee']: 1.3,
    JOINT_INDICES['right_knee']: 1.3,
    JOINT_INDICES['left_ankle']: 1.2,
    JOINT_INDICES['right_ankle']: 1.2,
}

# Dictionary mapping dance styles to their weights
DANCE_STYLE_WEIGHTS = {
    'ballet': BALLET_WEIGHTS,
    'hiphop': HIPHOP_WEIGHTS,
    'ballroom': BALLROOM_WEIGHTS,
    'contemporary': CONTEMPORARY_WEIGHTS,
    'latin': LATIN_WEIGHTS,
    'tap': TAP_WEIGHTS,
    'jazz': JAZZ_WEIGHTS
} 