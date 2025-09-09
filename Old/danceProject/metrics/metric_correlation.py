from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

class CorrelationMetric:
    """Correlation-based metric for comparing skeletal joint sequences."""
    
    def __init__(self, 
                 n_workers: int = 4,
                 min_correlation: float = -1.0):
        """
        Initialize correlation metric.
        
        Args:
            n_workers: Number of parallel workers for batch processing
            min_correlation: Minimum correlation value to consider (default: -1.0)
        """
        self.n_workers = n_workers
        self.min_correlation = min_correlation
        
    def _compute_joint_correlation(self, 
                                 traj1: np.ndarray, 
                                 traj2: np.ndarray) -> float:
        """
        Compute Pearson correlation between two joint trajectories.
        
        Args:
            traj1: First trajectory of shape (time_steps, 3)
            traj2: Second trajectory of shape (time_steps, 3)
            
        Returns:
            Average correlation across x, y, z coordinates
        """
        if traj1.shape != traj2.shape:
            return 0.0
            
        # Compute correlation for each dimension
        correlations = []
        for dim in range(3):
            try:
                corr, _ = pearsonr(traj1[:, dim], traj2[:, dim])
                correlations.append(max(corr, self.min_correlation))
            except:
                correlations.append(0.0)
                
        return float(np.mean(correlations))
    
    def _compute_sequence_correlation(self, 
                                   seq1: np.ndarray, 
                                   seq2: np.ndarray) -> float:
        """
        Compute average correlation across all joints.
        
        Args:
            seq1: First sequence of shape (time_steps, num_joints, 3)
            seq2: Second sequence of shape (time_steps, num_joints, 3)
            
        Returns:
            Average correlation across all joints
        """
        if seq1.shape != seq2.shape:
            return 0.0
            
        # Compute correlation for each joint
        joint_correlations = []
        for joint in range(seq1.shape[1]):
            corr = self._compute_joint_correlation(
                seq1[:, joint, :],
                seq2[:, joint, :]
            )
            joint_correlations.append(corr)
            
        return float(np.mean(joint_correlations))
    
    def compute_similarity(self, 
                         seq1: np.ndarray, 
                         seq2: np.ndarray) -> float:
        """
        Compute normalized similarity score between two sequences.
        
        Args:
            seq1: First sequence of shape (time_steps, num_joints, 3)
            seq2: Second sequence of shape (time_steps, num_joints, 3)
            
        Returns:
            Normalized similarity score (0-1)
        """
        # Input validation
        if seq1.shape[1:] != seq2.shape[1:]:
            raise ValueError("Sequences must have the same number of joints and dimensions")
            
        # Compute average correlation
        correlation = self._compute_sequence_correlation(seq1, seq2)
        
        # Convert to similarity score (0-1)
        # Map [-1, 1] to [0, 1]
        similarity = (correlation + 1.0) / 2.0
        return similarity
    
    def compute_batch_similarity(self, 
                               reference_seq: np.ndarray,
                               query_seqs: List[np.ndarray]) -> List[float]:
        """
        Compute similarity scores between a reference sequence and multiple query sequences in parallel.
        
        Args:
            reference_seq: Reference sequence of shape (time_steps, num_joints, 3)
            query_seqs: List of query sequences, each of shape (time_steps, num_joints, 3)
            
        Returns:
            List of similarity scores (0-1)
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Create partial function with reference sequence
            compute_partial = partial(self.compute_similarity, reference_seq)
            
            # Submit all tasks
            future_to_seq = {
                executor.submit(compute_partial, seq): i 
                for i, seq in enumerate(query_seqs)
            }
            
            # Collect results in order
            results = [0.0] * len(query_seqs)
            for future in as_completed(future_to_seq):
                idx = future_to_seq[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing sequence {idx}: {e}")
                    results[idx] = 0.0
                    
            return results
    
    def compute_pairwise_similarity(self, 
                                  sequences: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise similarity scores between all sequences in parallel.
        
        Args:
            sequences: List of sequences, each of shape (time_steps, num_joints, 3)
            
        Returns:
            Matrix of similarity scores (n_sequences x n_sequences)
        """
        n = len(sequences)
        similarity_matrix = np.zeros((n, n))
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Create tasks for upper triangular matrix
            futures = []
            for i in range(n):
                for j in range(i + 1, n):
                    future = executor.submit(
                        self.compute_similarity,
                        sequences[i],
                        sequences[j]
                    )
                    futures.append((future, i, j))
            
            # Collect results
            for future, i, j in futures:
                try:
                    similarity = future.result()
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Symmetric matrix
                except Exception as e:
                    print(f"Error processing sequences {i} and {j}: {e}")
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
                    
            # Set diagonal to 1.0 (self-similarity)
            np.fill_diagonal(similarity_matrix, 1.0)
            
        return similarity_matrix 