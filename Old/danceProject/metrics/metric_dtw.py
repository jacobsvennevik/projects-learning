from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

class DTWMetric:
    """Dynamic Time Warping metric for comparing skeletal joint sequences."""
    
    def __init__(self, 
                 radius: int = 10,
                 normalize: bool = True,
                 smooth_window: int = 5,
                 smooth_order: int = 2,
                 n_workers: int = 4):
        """
        Initialize DTW metric.
        
        Args:
            radius: Sakoe-Chiba bandwidth constraint
            normalize: Whether to normalize the final distance
            smooth_window: Window size for Savitzky-Golay smoothing
            smooth_order: Polynomial order for smoothing
            n_workers: Number of parallel workers for batch processing
        """
        self.radius = radius
        self.normalize = normalize
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order
        self.n_workers = n_workers
        
    def _smooth_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to the sequence.
        
        Args:
            sequence: Array of shape (time_steps, num_joints, 3)
            
        Returns:
            Smoothed sequence
        """
        if sequence.shape[0] < self.smooth_window:
            return sequence
            
        # Reshape for smoothing
        original_shape = sequence.shape
        reshaped = sequence.reshape(original_shape[0], -1)
        
        # Apply smoothing
        smoothed = savgol_filter(reshaped, 
                               window_length=self.smooth_window,
                               polyorder=self.smooth_order,
                               axis=0)
        
        return smoothed.reshape(original_shape)
    
    def _compute_joint_distances(self, 
                               seq1: np.ndarray, 
                               seq2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between joints in two sequences.
        
        Args:
            seq1: First sequence of shape (time_steps1, num_joints, 3)
            seq2: Second sequence of shape (time_steps2, num_joints, 3)
            
        Returns:
            Distance matrix of shape (time_steps1, time_steps2)
        """
        t1, n_joints, _ = seq1.shape
        t2, _, _ = seq2.shape
        
        # Reshape for distance computation
        seq1_flat = seq1.reshape(t1, n_joints * 3)
        seq2_flat = seq2.reshape(t2, n_joints * 3)
        
        # Compute pairwise distances
        distances = cdist(seq1_flat, seq2_flat, metric='euclidean')
        return distances
    
    def _compute_dtw_matrix(self, 
                          distances: np.ndarray, 
                          radius: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute DTW matrix and backtracking matrix.
        
        Args:
            distances: Distance matrix
            radius: Optional bandwidth constraint
            
        Returns:
            Tuple of (DTW matrix, backtracking matrix)
        """
        if radius is None:
            radius = self.radius
            
        n, m = distances.shape
        dtw = np.full((n + 1, m + 1), np.inf)
        backtrack = np.zeros((n + 1, m + 1), dtype=int)
        
        # Initialize first row and column
        dtw[0, 0] = 0
        for i in range(1, n + 1):
            dtw[i, 0] = np.inf
        for j in range(1, m + 1):
            dtw[0, j] = np.inf
            
        # Fill DTW matrix with constraints
        for i in range(1, n + 1):
            for j in range(max(1, i - radius), min(m + 1, i + radius + 1)):
                cost = distances[i - 1, j - 1]
                dtw[i, j] = cost + min(
                    dtw[i - 1, j],     # insertion
                    dtw[i, j - 1],     # deletion
                    dtw[i - 1, j - 1]  # match
                )
                
                # Store backtracking information
                if dtw[i, j] == cost + dtw[i - 1, j]:
                    backtrack[i, j] = 1  # insertion
                elif dtw[i, j] == cost + dtw[i, j - 1]:
                    backtrack[i, j] = 2  # deletion
                else:
                    backtrack[i, j] = 3  # match
                    
        return dtw, backtrack
    
    def _get_warping_path(self, 
                         dtw: np.ndarray, 
                         backtrack: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract the optimal warping path from the backtracking matrix.
        
        Args:
            dtw: DTW matrix
            backtrack: Backtracking matrix
            
        Returns:
            List of (i, j) pairs representing the warping path
        """
        path = []
        i, j = dtw.shape[0] - 1, dtw.shape[1] - 1
        
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))
            if backtrack[i, j] == 1:
                i -= 1
            elif backtrack[i, j] == 2:
                j -= 1
            else:
                i -= 1
                j -= 1
                
        return list(reversed(path))
    
    def _normalize_distance(self, 
                          dtw: np.ndarray, 
                          path: List[Tuple[int, int]]) -> float:
        """
        Normalize the DTW distance by path length.
        
        Args:
            dtw: DTW matrix
            path: Optimal warping path
            
        Returns:
            Normalized distance
        """
        if not path:
            return 0.0
            
        total_cost = sum(dtw[i + 1, j + 1] for i, j in path)
        return total_cost / len(path)
    
    def compute_similarity(self, 
                         seq1: np.ndarray, 
                         seq2: np.ndarray) -> float:
        """
        Compute normalized similarity score between two sequences.
        
        Args:
            seq1: First sequence of shape (time_steps1, num_joints, 3)
            seq2: Second sequence of shape (time_steps2, num_joints, 3)
            
        Returns:
            Normalized similarity score (0-1)
        """
        # Input validation
        if seq1.shape[1:] != seq2.shape[1:]:
            raise ValueError("Sequences must have the same number of joints and dimensions")
            
        # Smooth sequences if needed
        if self.smooth_window > 1:
            seq1 = self._smooth_sequence(seq1)
            seq2 = self._smooth_sequence(seq2)
            
        # Compute distance matrix
        distances = self._compute_joint_distances(seq1, seq2)
        
        # Compute DTW matrix and get warping path
        dtw, backtrack = self._compute_dtw_matrix(distances)
        path = self._get_warping_path(dtw, backtrack)
        
        # Compute normalized distance
        distance = self._normalize_distance(dtw, path)
        
        # Convert to similarity score (0-1)
        if self.normalize:
            # Use exponential decay for better scaling
            similarity = np.exp(-distance)
            return float(similarity)
        else:
            return float(distance)
    
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