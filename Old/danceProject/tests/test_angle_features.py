import pytest
import numpy as np
from features.angle_features import AngleFeatureExtractor

@pytest.fixture
def mock_joints():
    """Create mock joint data for testing."""
    # Create a simple pose with 10 frames and 17 joints
    n_frames = 10
    n_joints = 17
    joints = np.zeros((n_frames, n_joints, 3))
    
    # Create a simple wave-like motion
    for frame in range(n_frames):
        # Base positions
        joints[frame] = np.array([
            [0, 0, 0],    # nose
            [0, 1, 0],    # left_eye
            [0, 1, 0],    # right_eye
            [0, 2, 0],    # left_ear
            [0, 2, 0],    # right_ear
            [0, 3, 0],    # left_shoulder
            [0, 3, 0],    # right_shoulder
            [0, 4, 0],    # left_elbow
            [0, 4, 0],    # right_elbow
            [0, 5, 0],    # left_wrist
            [0, 5, 0],    # right_wrist
            [0, 6, 0],    # left_hip
            [0, 6, 0],    # right_hip
            [0, 7, 0],    # left_knee
            [0, 7, 0],    # right_knee
            [0, 8, 0],    # left_ankle
            [0, 8, 0]     # right_ankle
        ])
        
        # Add wave-like motion
        joints[frame, [5, 6, 8, 9]] += [0, 0, np.sin(frame/2)]  # Move arms
    
    return joints

def test_angle_feature_extractor_initialization():
    """Test that the angle feature extractor initializes correctly."""
    extractor = AngleFeatureExtractor()
    
    # Check angle pairs
    assert len(extractor.angle_pairs) == 8  # 8 different angle measurements
    
    # Check joint indices
    assert len(extractor.joint_indices) == 17  # 17 joints
    assert extractor.joint_indices['nose'] == 0
    assert extractor.joint_indices['right_ankle'] == 16

def test_angle_feature_extractor_compute_angle():
    """Test angle computation between three points."""
    extractor = AngleFeatureExtractor()
    
    # Test zero angle
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    
    angle = extractor._compute_angle(p1, p2, p3)
    assert np.isclose(angle, 0.0)
    
    # Test right angle
    p1 = np.array([0, 0, 0])  # Origin
    p2 = np.array([1, 0, 0])  # Point on x-axis
    p3 = np.array([1, 1, 0])  # Point above p2
    
    angle = extractor._compute_angle(p1, p2, p3)
    assert np.isclose(angle, np.pi/2)

def test_angle_feature_extractor_extract_features(mock_joints):
    """Test feature extraction from joint data."""
    extractor = AngleFeatureExtractor()
    
    # Extract features
    features = extractor.extract_features(mock_joints)
    
    # Check output structure
    assert isinstance(features, dict)
    assert 'angles' in features
    assert 'angles_3d' in features
    
    # Check output shapes
    n_frames = mock_joints.shape[0]
    n_angles = len(extractor.angle_pairs)
    assert features['angles'].shape == (n_frames, n_angles)
    assert features['angles_3d'].shape == (n_frames, n_angles, 3)
    
    # Check feature values
    assert np.all(features['angles'] >= 0)  # All angles should be non-negative
    assert np.all(features['angles'] <= np.pi)  # All angles should be <= π
    assert np.all(features['angles_3d'] >= 0)  # All 3D angles should be non-negative
    assert np.all(features['angles_3d'] <= np.pi)  # All 3D angles should be <= π

def test_angle_feature_extractor_feature_names():
    """Test feature name generation."""
    extractor = AngleFeatureExtractor()
    names = extractor.get_feature_names()
    
    # Check number of names
    assert len(names) == len(extractor.angle_pairs)
    
    # Check name format
    for name in names:
        assert name.startswith("angle_")
        assert name.count("_") == 2  # Should have 2 underscores

def test_angle_feature_extractor_empty_input():
    """Test handling of empty input."""
    extractor = AngleFeatureExtractor()
    empty_data = np.array([]).reshape(0, 17, 3)
    
    with pytest.raises(ValueError, match="Empty input array"):
        extractor.extract_features(empty_data)

def test_angle_feature_extractor_invalid_input_shape():
    """Test handling of invalid input shape."""
    extractor = AngleFeatureExtractor()
    invalid_data = np.random.rand(10, 15, 2)  # Wrong number of joints
    
    with pytest.raises(ValueError, match="Expected 17 joints"):
        extractor.extract_features(invalid_data)

def test_angle_feature_extractor_nan_handling():
    """Test handling of NaN values in input."""
    extractor = AngleFeatureExtractor()
    
    # Create data with NaN values
    data = np.random.rand(3, 17, 3)
    data[0, 0] = np.nan  # Add NaN to first joint in first frame
    
    features = extractor.extract_features(data)
    
    # Check that NaN values are handled gracefully
    assert not np.any(np.isnan(features['angles']))
    assert not np.any(np.isnan(features['angles_3d']))

def test_angle_feature_extractor_parallel_processing():
    """Test that parallel processing produces same results as sequential."""
    extractor = AngleFeatureExtractor()
    
    # Create a larger dataset
    data = np.random.rand(10, 17, 3)
    
    # Extract features with and without parallel processing
    features_seq = extractor.extract_features(data, n_workers=1)
    features_par = extractor.extract_features(data, n_workers=4)
    
    # Check that results are identical
    np.testing.assert_array_equal(features_seq['angles'], features_par['angles'])
    np.testing.assert_array_equal(features_seq['angles_3d'], features_par['angles_3d'])

@pytest.fixture
def synthetic_joint_data():
    """Create synthetic joint data for testing."""
    # Create a simple sequence with 3 frames
    n_frames = 3
    n_joints = 17  # Number of joints in the skeleton
    
    # Create synthetic 3D coordinates
    # For simplicity, use a simple pattern where joints move in a predictable way
    data = np.zeros((n_frames, n_joints, 3))
    
    # Frame 0: Initial positions
    data[0] = np.array([
        [0, 0, 0],    # nose
        [0, 1, 0],    # left_eye
        [0, 1, 0],    # right_eye
        [0, 2, 0],    # left_ear
        [0, 2, 0],    # right_ear
        [0, 3, 0],    # left_shoulder
        [0, 3, 0],    # right_shoulder
        [0, 4, 0],    # left_elbow
        [0, 4, 0],    # right_elbow
        [0, 5, 0],    # left_wrist
        [0, 5, 0],    # right_wrist
        [0, 6, 0],    # left_hip
        [0, 6, 0],    # right_hip
        [0, 7, 0],    # left_knee
        [0, 7, 0],    # right_knee
        [0, 8, 0],    # left_ankle
        [0, 8, 0]     # right_ankle
    ])
    
    # Frame 1: Move arms up
    data[1] = data[0].copy()
    data[1, [5, 6, 8, 9]] = data[1, [5, 6, 8, 9]] + [0, 0, 1]  # Move elbows and wrists up
    
    # Frame 2: Move arms down
    data[2] = data[0].copy()
    data[2, [5, 6, 8, 9]] = data[2, [5, 6, 8, 9]] + [0, 0, -1]  # Move elbows and wrists down
    
    return data

@pytest.fixture
def feature_extractor():
    """Create an instance of AngleFeatureExtractor."""
    return AngleFeatureExtractor()

def test_initialization(feature_extractor):
    """Test that the feature extractor initializes correctly."""
    assert feature_extractor is not None
    assert feature_extractor.angle_pairs is not None
    assert len(feature_extractor.angle_pairs) > 0

def test_extract_features_shape(feature_extractor, synthetic_joint_data):
    """Test that extracted features have the correct shape."""
    features = feature_extractor.extract_features(synthetic_joint_data)
    
    # Check that features is a dictionary
    assert isinstance(features, dict)
    
    # Check that all required keys are present
    required_keys = ['angles', 'angles_3d']
    assert all(key in features for key in required_keys)
    
    # Check shapes
    n_frames = synthetic_joint_data.shape[0]
    n_angles = len(feature_extractor.angle_pairs)
    
    assert features['angles'].shape == (n_frames, n_angles)
    assert features['angles_3d'].shape == (n_frames, n_angles, 3)

def test_angle_calculation(feature_extractor, synthetic_joint_data):
    """Test that angle calculations are correct for known configurations."""
    features = feature_extractor.extract_features(synthetic_joint_data)
    
    # Test frame 0 (initial position)
    angles_0 = features['angles'][0]
    angles_3d_0 = features['angles_3d'][0]
    
    # Check that angles are within valid range
    assert np.all(angles_0 >= 0)
    assert np.all(angles_0 <= 180)
    assert np.all(angles_3d_0 >= 0)
    assert np.all(angles_3d_0 <= 180)
    
    # Test frame 1 (arms up)
    angles_1 = features['angles'][1]
    angles_3d_1 = features['angles_3d'][1]
    
    # Check that arm angles have changed
    arm_angle_indices = [i for i, (j1, j2, j3) in enumerate(feature_extractor.angle_pairs)
                        if 'elbow' in j1 or 'elbow' in j2 or 'elbow' in j3]
    
    assert not np.array_equal(angles_0[arm_angle_indices], angles_1[arm_angle_indices])
    assert not np.array_equal(angles_3d_0[arm_angle_indices], angles_3d_1[arm_angle_indices])

def test_angle_consistency(feature_extractor, synthetic_joint_data):
    """Test that 2D and 3D angles are consistent for planar movements."""
    features = feature_extractor.extract_features(synthetic_joint_data)
    
    # For planar movements (in XY plane), 2D and 3D angles should be similar
    angles_2d = features['angles']
    angles_3d = features['angles_3d'][:, :, 2]  # Take Z component of 3D angles
    
    # Check that angles are similar (within small tolerance)
    np.testing.assert_allclose(angles_2d, angles_3d, rtol=1e-5, atol=1e-5)

def test_angle_pairs_consistency(feature_extractor):
    """Test that angle pairs are consistent and valid."""
    for j1, j2, j3 in feature_extractor.angle_pairs:
        # Check that all joint names are valid
        assert j1 in feature_extractor.joint_indices
        assert j2 in feature_extractor.joint_indices
        assert j3 in feature_extractor.joint_indices
        
        # Check that middle joint is different from endpoints
        assert j2 != j1
        assert j2 != j3
        assert j1 != j3

def test_nan_handling(feature_extractor):
    """Test handling of NaN values in input."""
    # Create data with NaN values
    data = np.random.rand(3, 17, 3)
    data[0, 0] = np.nan  # Add NaN to first joint in first frame
    
    features = feature_extractor.extract_features(data)
    
    # Check that NaN values are handled gracefully
    assert not np.any(np.isnan(features['angles']))
    assert not np.any(np.isnan(features['angles_3d']))

def test_parallel_processing(feature_extractor, synthetic_joint_data):
    """Test that parallel processing produces same results as sequential."""
    # Create a larger dataset
    large_data = np.tile(synthetic_joint_data, (10, 1, 1))
    
    # Extract features with and without parallel processing
    features_seq = feature_extractor.extract_features(large_data, n_workers=1)
    features_par = feature_extractor.extract_features(large_data, n_workers=4)
    
    # Check that results are identical
    np.testing.assert_array_equal(features_seq['angles'], features_par['angles'])
    np.testing.assert_array_equal(features_seq['angles_3d'], features_par['angles_3d']) 