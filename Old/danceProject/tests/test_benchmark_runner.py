import pytest
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from benchmarking.benchmark_runner import BenchmarkRunner
from typing import Dict, List

@pytest.fixture
def mock_dataset(tmp_path) -> Dict:
    """Create a mock dataset for testing."""
    # Create mock sequences
    n_frames = 10
    n_joints = 17
    n_ref_sequences = 2
    n_query_sequences = 3
    
    # Create reference sequences
    ref_sequences = {}
    for i in range(n_ref_sequences):
        ref_sequences[f"ref_{i}"] = {
            "sequence": np.random.randn(n_frames, n_joints, 3).tolist(),
            "label": f"class_{i % 2}"
        }
    
    # Create query sequences
    query_sequences = {}
    for i in range(n_query_sequences):
        query_sequences[f"query_{i}"] = {
            "sequence": np.random.randn(n_frames, n_joints, 3).tolist(),
            "label": f"class_{i % 2}"
        }
    
    # Create dataset dictionary
    dataset = {
        "reference_sequences": ref_sequences,
        "query_sequences": query_sequences
    }
    
    # Save dataset to temporary directory
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    
    # Save skeleton dataset
    with open(dataset_path / "skeleton_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    # Save angles dataset
    with open(dataset_path / "angles_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    return dataset_path

def test_benchmark_runner_initialization(mock_dataset):
    """Test BenchmarkRunner initialization."""
    # Test valid initialization
    runner = BenchmarkRunner(str(mock_dataset), feature_type="skeleton")
    assert runner.dataset_path == str(mock_dataset)
    assert runner.feature_type == "skeleton"
    assert runner.n_workers == 4  # default
    assert runner.dance_style is None
    
    # Test invalid feature type
    with pytest.raises(ValueError, match="Invalid feature type"):
        BenchmarkRunner(str(mock_dataset), feature_type="invalid")
    
    # Test invalid number of workers
    with pytest.raises(ValueError, match="Number of workers must be positive"):
        BenchmarkRunner(str(mock_dataset), n_workers=0)

def test_benchmark_runner_load_dataset(mock_dataset):
    """Test dataset loading functionality."""
    runner = BenchmarkRunner(str(mock_dataset))
    
    # Test loading skeleton dataset
    dataset = runner._load_dataset()
    assert "reference_sequences" in dataset
    assert "query_sequences" in dataset
    assert len(dataset["reference_sequences"]) == 2
    assert len(dataset["query_sequences"]) == 3
    
    # Test loading angles dataset
    runner = BenchmarkRunner(str(mock_dataset), feature_type="angles")
    dataset = runner._load_dataset()
    assert "reference_sequences" in dataset
    assert "query_sequences" in dataset

def test_benchmark_runner_compute_metric_scores(mock_dataset):
    """Test metric score computation."""
    runner = BenchmarkRunner(str(mock_dataset))
    
    # Create test sequences
    ref_seq = np.random.randn(10, 17, 3)
    query_seq = np.random.randn(10, 17, 3)
    
    # Test each metric
    for metric_name in runner.metrics:
        scores, confidence = runner._compute_metric_scores(
            metric_name,
            ref_seq,
            [query_seq]
        )
        
        assert isinstance(scores, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert len(scores) == 1
        assert len(confidence) == 1
        assert 0 <= scores[0] <= 1
        assert 0 <= confidence[0] <= 1

def test_benchmark_runner_run_benchmark(mock_dataset):
    """Test running the benchmark."""
    runner = BenchmarkRunner(str(mock_dataset))
    results = runner.run_benchmark()
    
    # Check results DataFrame
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert all(col in results.columns for col in [
        "reference", "query", "metric", "score", "confidence", "true_label"
    ])
    
    # Check output files
    results_dir = mock_dataset / "results"
    assert (results_dir / "detailed_results.csv").exists()
    assert (results_dir / "summary.csv").exists()

def test_benchmark_runner_error_handling(mock_dataset):
    """Test error handling in BenchmarkRunner."""
    # Test missing dataset file
    with pytest.raises(FileNotFoundError):
        BenchmarkRunner(str(mock_dataset / "nonexistent"))
    
    # Test invalid dataset structure
    invalid_dataset = mock_dataset / "invalid"
    invalid_dataset.mkdir()
    with open(invalid_dataset / "skeleton_dataset.json", "w") as f:
        json.dump({"invalid": "structure"}, f)
    
    with pytest.raises(ValueError, match="No sequences found in dataset"):
        BenchmarkRunner(str(invalid_dataset))
    
    # Test missing label in query sequence
    invalid_dataset = mock_dataset / "missing_label"
    invalid_dataset.mkdir()
    dataset = {
        "reference_sequences": {"ref_0": {"sequence": np.random.randn(10, 17, 3).tolist()}},
        "query_sequences": {"query_0": {"sequence": np.random.randn(10, 17, 3).tolist()}}
    }
    with open(invalid_dataset / "skeleton_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    runner = BenchmarkRunner(str(invalid_dataset))
    with pytest.raises(KeyError, match="Missing label in query sequence"):
        runner.run_benchmark()
    
    # Test inconsistent sequence shapes
    invalid_dataset = mock_dataset / "inconsistent_shapes"
    invalid_dataset.mkdir()
    dataset = {
        "reference_sequences": {
            "ref_0": {
                "sequence": np.random.randn(10, 17, 3).tolist(),
                "label": "class_0"
            }
        },
        "query_sequences": {
            "query_0": {
                "sequence": np.random.randn(5, 17, 3).tolist(),  # Different length
                "label": "class_0"
            }
        }
    }
    with open(invalid_dataset / "skeleton_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    runner = BenchmarkRunner(str(invalid_dataset))
    with pytest.raises(ValueError, match="Inconsistent sequence lengths"):
        runner.run_benchmark()

def test_benchmark_runner_with_dance_style(mock_dataset):
    """Test BenchmarkRunner with dance style specification."""
    runner = BenchmarkRunner(str(mock_dataset), dance_style="ballet")
    results = runner.run_benchmark()
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    
    # Check that TLCC metric is using the correct dance style
    tlcc_results = results[results["metric"] == "tlcc"]
    assert len(tlcc_results) > 0

def test_benchmark_runner_dance_style(mock_dataset):
    """Test that the benchmark runner handles dance styles correctly."""
    # Create test sequences with non-zero baseline
    ref_seq = np.ones((10, 33, 3))  # Use all 33 joints from MediaPipe
    query_seq = ref_seq.copy()

    # Create a more realistic leg movement pattern
    # Legs and feet joints (indices from JOINT_INDICES)
    leg_joints = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # hip to foot indices
    arm_joints = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # shoulder to fingers

    # Create a smooth, periodic leg movement pattern
    t = np.linspace(0, 2*np.pi, 10)

    # Add significant leg movements (more weighted in ballet)
    for joint in leg_joints:
        # Add large movements in all dimensions for query sequence
        query_seq[:, joint, 0] = 1 + np.sin(t) * 1.5  # x-axis
        query_seq[:, joint, 1] = 1 + np.cos(t) * 1.5  # y-axis
        query_seq[:, joint, 2] = 1 + np.sin(2*t) * 1.5  # z-axis

        # Add similar but smaller movements to reference sequence
        ref_seq[:, joint, 0] = 1 + np.sin(t + np.pi/4) * 1.2  # x-axis (phase shift)
        ref_seq[:, joint, 1] = 1 + np.cos(t + np.pi/4) * 1.2  # y-axis (phase shift)
        ref_seq[:, joint, 2] = 1 + np.sin(2*t + np.pi/4) * 1.2  # z-axis (phase shift)

    # Add minimal arm movements for ballet
    for joint in arm_joints:
        # Add small movements in all dimensions for query sequence
        query_seq[:, joint, 0] = 1 + np.sin(t) * 0.2  # x-axis
        query_seq[:, joint, 1] = 1 + np.cos(t) * 0.2  # y-axis
        query_seq[:, joint, 2] = 1 + np.sin(2*t) * 0.2  # z-axis

        # Add similar small movements to reference sequence
        ref_seq[:, joint, 0] = 1 + np.sin(t + np.pi/6) * 0.15  # x-axis (small phase shift)
        ref_seq[:, joint, 1] = 1 + np.cos(t + np.pi/6) * 0.15  # y-axis (small phase shift)
        ref_seq[:, joint, 2] = 1 + np.sin(2*t + np.pi/6) * 0.15  # z-axis (small phase shift)

    # Run benchmark with ballet style
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        dance_style="ballet",
        n_workers=2
    )

    # Modify dataset to use our test sequences
    runner.dataset = {
        'reference_sequences': {'ref_0': {'sequence': ref_seq.tolist()}},
        'query_sequences': {'query_0': {'sequence': query_seq.tolist(), 'label': 1}}
    }

    results = runner.run_benchmark()
    ballet_tlcc_score = results[results['metric'] == 'tlcc']['score'].iloc[0]

    # Create new sequences for hip-hop with emphasis on arm movements
    ref_seq_hiphop = np.ones((10, 33, 3))
    query_seq_hiphop = ref_seq_hiphop.copy()

    # Add minimal leg movements for hip-hop
    for joint in leg_joints:
        # Add small movements in all dimensions for query sequence
        query_seq_hiphop[:, joint, 0] = 1 + np.sin(t) * 0.2  # x-axis
        query_seq_hiphop[:, joint, 1] = 1 + np.cos(t) * 0.2  # y-axis
        query_seq_hiphop[:, joint, 2] = 1 + np.sin(2*t) * 0.2  # z-axis

        # Add similar small movements to reference sequence
        ref_seq_hiphop[:, joint, 0] = 1 + np.sin(t + np.pi/6) * 0.15  # x-axis
        ref_seq_hiphop[:, joint, 1] = 1 + np.cos(t + np.pi/6) * 0.15  # y-axis
        ref_seq_hiphop[:, joint, 2] = 1 + np.sin(2*t + np.pi/6) * 0.15  # z-axis

    # Add significant arm movements for hip-hop
    for joint in arm_joints:
        # Add large movements in all dimensions for query sequence
        query_seq_hiphop[:, joint, 0] = 1 + np.sin(t) * 1.5  # x-axis
        query_seq_hiphop[:, joint, 1] = 1 + np.cos(t) * 1.5  # y-axis
        query_seq_hiphop[:, joint, 2] = 1 + np.sin(2*t) * 1.5  # z-axis

        # Add similar but smaller movements to reference sequence
        ref_seq_hiphop[:, joint, 0] = 1 + np.sin(t + np.pi/4) * 1.2  # x-axis
        ref_seq_hiphop[:, joint, 1] = 1 + np.cos(t + np.pi/4) * 1.2  # y-axis
        ref_seq_hiphop[:, joint, 2] = 1 + np.sin(2*t + np.pi/4) * 1.2  # z-axis

    # Run benchmark with hiphop style
    runner2 = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        dance_style="hiphop",
        n_workers=2
    )

    # Use hip-hop specific dataset
    runner2.dataset = {
        'reference_sequences': {'ref_0': {'sequence': ref_seq_hiphop.tolist()}},
        'query_sequences': {'query_0': {'sequence': query_seq_hiphop.tolist(), 'label': 1}}
    }

    results2 = runner2.run_benchmark()
    hiphop_tlcc_score = results2[results2['metric'] == 'tlcc']['score'].iloc[0]

    # Results should be different due to different joint weights
    # Ballet should have higher scores for leg movements
    assert ballet_tlcc_score > hiphop_tlcc_score, (
        f"Ballet TLCC score ({ballet_tlcc_score}) should be higher than Hip-hop TLCC score ({hiphop_tlcc_score}) "
        f"for leg movements. This might indicate an issue with joint weights or metric computation."
    )

def test_benchmark_runner_output_shape(mock_dataset):
    """Test that the benchmark runner produces correctly shaped output."""
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        n_workers=2
    )
    
    results = runner.run_benchmark()
    
    # Check DataFrame shape
    n_ref = len(runner.dataset["reference_sequences"])
    n_query = len(runner.dataset["query_sequences"])
    n_metrics = len(runner.metrics)
    
    assert results.shape == (n_ref * n_query * n_metrics, 6)  # 6 columns: reference, query, metric, score, confidence, true_label

def test_benchmark_runner_metrics(mock_dataset):
    """Test that the benchmark runner computes correct metrics."""
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        n_workers=2
    )
    
    results = runner.run_benchmark()
    metrics_df = runner.compute_metrics(results)
    
    # Check metrics DataFrame shape and columns
    assert metrics_df.shape == (len(runner.metrics), 3)  # 3 columns: accuracy, avg_confidence, weighted_accuracy
    assert all(col in metrics_df.columns for col in ['accuracy', 'avg_confidence', 'weighted_accuracy'])
    
    # Check that metrics are within valid ranges
    assert all(metrics_df["accuracy"] >= 0)
    assert all(metrics_df["accuracy"] <= 1)
    assert all(metrics_df["avg_confidence"] >= 0)
    assert all(metrics_df["avg_confidence"] <= 1)
    assert all(metrics_df["weighted_accuracy"] >= 0)
    assert all(metrics_df["weighted_accuracy"] <= 1)

def test_benchmark_runner_ranking(mock_dataset):
    """Test that the benchmark runner produces correct ranking of metrics."""
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        n_workers=2
    )
    
    results = runner.run_benchmark()
    metrics_df = runner.compute_metrics(results)
    
    # Sort by weighted accuracy
    ranked_metrics = metrics_df.sort_values("weighted_accuracy", ascending=False)
    
    # Check that metrics are ranked correctly
    assert ranked_metrics.index[0] in runner.metrics.keys()
    
    # Check that weighted accuracy is non-increasing
    assert all(ranked_metrics["weighted_accuracy"].diff().dropna() <= 0)

def test_benchmark_runner_output_files(mock_dataset):
    """Test that the benchmark runner creates output files."""
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        n_workers=2
    )
    
    results = runner.run_benchmark()
    metrics_df = runner.compute_metrics(results)
    runner.print_results(metrics_df)
    
    # Check that output file exists
    output_file = mock_dataset / "benchmark_results_skeleton.csv"
    assert output_file.exists()
    
    # Check that output file contains correct data
    saved_df = pd.read_csv(output_file, index_col=0)  # Read with index column
    pd.testing.assert_frame_equal(metrics_df, saved_df)

def test_benchmark_runner_parallel_processing(mock_dataset):
    """Test that parallel processing produces consistent results."""
    # Run with different numbers of workers
    runners = [
        BenchmarkRunner(
            dataset_path=str(mock_dataset),
            feature_type="skeleton",
            n_workers=n
        ) for n in [1, 2, 4]
    ]
    
    results = []
    for runner in runners:
        r = runner.run_benchmark()
        m = runner.compute_metrics(r)
        results.append(m)
    
    # Check that results are consistent across different worker counts
    for i in range(1, len(results)):
        pd.testing.assert_frame_equal(results[0], results[i])

def test_benchmark_runner_empty_dataset(tmp_path):
    """Test that the benchmark runner handles empty datasets correctly."""
    # Create empty dataset
    dataset_dir = tmp_path / "empty_dataset"
    dataset_dir.mkdir()
    
    # Create empty dataset file
    empty_dataset = {
        "reference_sequences": {
            "ref_0": {
                "sequence": []  # Empty sequence
            }
        },
        "query_sequences": {
            "query_0": {
                "sequence": [],  # Empty sequence
                "label": 1
            }
        }
    }
    
    with open(dataset_dir / "skeleton_dataset.json", "w") as f:
        json.dump(empty_dataset, f)
    
    # Test that initialization raises ValueError for empty dataset
    with pytest.raises(ValueError, match="No sequences found in dataset"):
        BenchmarkRunner(
            dataset_path=str(dataset_dir),
            feature_type="skeleton",
            n_workers=2
        )

def test_benchmark_runner_missing_labels(mock_dataset):
    """Test that the benchmark runner handles missing labels correctly."""
    # Modify dataset to remove some labels
    with open(mock_dataset / "skeleton_dataset.json", "r") as f:
        dataset = json.load(f)
    
    # Remove label from first query sequence
    dataset["query_sequences"]["query_0"].pop("label")
    
    with open(mock_dataset / "skeleton_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        n_workers=2
    )
    
    with pytest.raises(KeyError, match="Missing label in query sequence"):
        runner.run_benchmark()

def test_benchmark_runner_invalid_sequence_length(mock_dataset):
    """Test that the benchmark runner handles sequences of different lengths."""
    # Modify dataset to have different sequence lengths
    with open(mock_dataset / "skeleton_dataset.json", "r") as f:
        dataset = json.load(f)
    
    # Make first query sequence shorter
    sequence = dataset["query_sequences"]["query_0"]["sequence"]
    dataset["query_sequences"]["query_0"]["sequence"] = sequence[:5]  # Only 5 frames
    
    with open(mock_dataset / "skeleton_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        n_workers=2
    )
    
    with pytest.raises(ValueError, match="Inconsistent sequence lengths"):
        runner.run_benchmark()

def test_benchmark_runner_invalid_joint_count(mock_dataset):
    """Test that the benchmark runner handles invalid joint counts."""
    # Modify dataset to have incorrect number of joints
    with open(mock_dataset / "skeleton_dataset.json", "r") as f:
        dataset = json.load(f)
    
    # Modify first query sequence to have 16 joints instead of 17
    sequence = dataset["query_sequences"]["query_0"]["sequence"]
    for frame in sequence:
        frame.pop()  # Remove last joint
    
    with open(mock_dataset / "skeleton_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    runner = BenchmarkRunner(
        dataset_path=str(mock_dataset),
        feature_type="skeleton",
        n_workers=2
    )
    
    with pytest.raises(ValueError, match="Inconsistent number of joints"):
        runner.run_benchmark()

def test_benchmark_runner_invalid_feature_type(mock_dataset):
    """Test that the benchmark runner handles invalid feature types."""
    with pytest.raises(ValueError, match="Invalid feature type"):
        BenchmarkRunner(
            dataset_path=str(mock_dataset),
            feature_type="invalid_type",
            n_workers=2
        )

def test_benchmark_runner_invalid_dance_style(mock_dataset):
    """Test that the benchmark runner handles invalid dance styles."""
    with pytest.raises(ValueError, match="Invalid dance style"):
        BenchmarkRunner(
            dataset_path=str(mock_dataset),
            feature_type="skeleton",
            dance_style="invalid_style",
            n_workers=2
        )

def test_benchmark_runner_negative_workers(mock_dataset):
    """Test that the benchmark runner handles invalid worker counts."""
    with pytest.raises(ValueError, match="Number of workers must be positive"):
        BenchmarkRunner(
            dataset_path=str(mock_dataset),
            feature_type="skeleton",
            n_workers=-1
        ) 