from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from .joint_weights import DANCE_STYLE_WEIGHTS, JOINT_INDICES

class TLCCMetric:
    """Time Lagged Cross Correlation metric for dance pose comparison."""
    
    def __init__(self,
                 max_lag: int = 10,
                 min_correlation: float = 0.0,
                 n_workers: int = 4,
                 dance_style: Optional[str] = None):
        """
        Initialize the TLCC metric.
        
        Args:
            max_lag: Maximum time lag to consider
            min_correlation: Minimum correlation threshold
            n_workers: Number of parallel workers
            dance_style: Optional dance style for joint weighting
        """
        self.max_lag = max_lag
        self.min_correlation = min_correlation
        self.n_workers = n_workers
        
        # Validate dance style
        if dance_style is not None and dance_style not in DANCE_STYLE_WEIGHTS:
            raise ValueError(f"Invalid dance style. Must be one of: {list(DANCE_STYLE_WEIGHTS.keys())}")
        self.dance_style = dance_style
    
    def _compute_lagged_correlation(self,
                                  x: np.ndarray,
                                  y: np.ndarray,
                                  lag: int) -> float:
        """
        Compute correlation between two sequences with a given lag.
        
        Args:
            x: First sequence
            y: Second sequence
            lag: Time lag to apply
            
        Returns:
            Correlation coefficient
        """
        if lag > 0:
            x = x[lag:]
            y = y[:-lag]
        else:
            x = x[:lag]
            y = y[-lag:]
        
        if len(x) < 2 or len(y) < 2:
            return 0.0
        
        # Compute correlation for each dimension
        correlations = []
        for i in range(x.shape[1]):
            corr, _ = pearsonr(x[:, i], y[:, i])
            correlations.append(corr)
        
        # Return mean correlation across dimensions
        mean_corr = np.mean(correlations)
        return max(mean_corr, self.min_correlation)
    
    def _find_optimal_lag(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Tuple[float, int]:
        """
        Find the optimal lag between two sequences.
        
        Args:
            x: First sequence
            y: Second sequence
            
        Returns:
            Tuple of (best correlation, optimal lag)
        """
        best_corr = -float('inf')
        best_lag = 0
        
        for lag in range(-self.max_lag, self.max_lag + 1):
            corr = self._compute_lagged_correlation(x, y, lag)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        
        return best_corr, best_lag
    
    def _compute_joint_tlcc(self,
                          joint1: np.ndarray,
                          joint2: np.ndarray) -> float:
        """
        Compute TLCC for a single joint pair.
        
        Args:
            joint1: First joint sequence
            joint2: Second joint sequence
            
        Returns:
            Best correlation score
        """
        correlation, _ = self._find_optimal_lag(joint1, joint2)
        return correlation
    
    def _compute_sequence_tlcc(self,
                             seq1: np.ndarray,
                             seq2: np.ndarray) -> float:
        """
        Compute TLCC for entire sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Average correlation score
        """
        n_joints = seq1.shape[1]
        correlations = []
        weights = []
        
        # Get joint weights if dance style is specified
        style_weights = None
        if self.dance_style:
            style_weights = DANCE_STYLE_WEIGHTS[self.dance_style]
        
        for i in range(n_joints):
            joint1 = seq1[:, i]
            joint2 = seq2[:, i]
            
            # Compute correlation for this joint
            correlation = self._compute_joint_tlcc(joint1, joint2)
            
            # Get weight for this joint
            weight = 1.0
            if style_weights is not None and i in style_weights:
                weight = style_weights[i]
            
            correlations.append(correlation)
            weights.append(weight)
        
        # Convert to numpy arrays for vectorized operations
        correlations = np.array(correlations)
        weights = np.array(weights)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Compute weighted average of correlations
        weighted_correlation = np.sum(correlations * weights)
        
        return weighted_correlation
    
    def compute_similarity(self,
                         seq1: np.ndarray,
                         seq2: np.ndarray) -> float:
        """
        Compute similarity between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Similarity score between 0 and 1
        """
        return self._compute_sequence_tlcc(seq1, seq2)
    
    def compute_batch_similarity(self,
                               reference_seq: np.ndarray,
                               query_seqs: List[np.ndarray]) -> Tuple[List[float], List[int]]:
        """
        Compute similarity between a reference sequence and multiple query sequences.
        
        Args:
            reference_seq: Reference sequence
            query_seqs: List of query sequences
            
        Returns:
            Tuple of (similarity scores, optimal lags)
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for query_seq in query_seqs:
                future = executor.submit(self.compute_similarity,
                                      reference_seq,
                                      query_seq)
                futures.append(future)
            
            scores = []
            lags = []  # Placeholder for lags
            for future in as_completed(futures):
                score = future.result()
                scores.append(score)
                lags.append(0)  # Placeholder lag value
            
            return scores, lags
    
    def compute_pairwise_similarity(self,
                                  sequences: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise similarity between multiple sequences.
        
        Args:
            sequences: List of sequences
            
        Returns:
            Matrix of pairwise similarities
        """
        n_seqs = len(sequences)
        similarity_matrix = np.zeros((n_seqs, n_seqs))
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(n_seqs):
                for j in range(i + 1, n_seqs):
                    future = executor.submit(self.compute_similarity,
                                          sequences[i],
                                          sequences[j])
                    futures.append((i, j, future))
            
            for i, j, future in futures:
                similarity = future.result()
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix 