import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path

from metrics.metric_dtw import DTWMetric
from metrics.metric_correlation import CorrelationMetric
from metrics.metric_tlcc import TLCCMetric
from metrics.joint_weights import DANCE_STYLE_WEIGHTS

class BenchmarkRunner:
    """Runs benchmark comparisons between different metrics."""
    
    def __init__(self,
                 dataset_path: str,
                 feature_type: str = "skeleton",
                 n_workers: int = 4,
                 dance_style: Optional[str] = None):
        """
        Initialize the benchmark runner.
        
        Args:
            dataset_path: Path to the dataset directory
            feature_type: Type of features to use ("skeleton" or "angles")
            n_workers: Number of parallel workers
            dance_style: Optional dance style for joint weighting
        """
        if n_workers <= 0:
            raise ValueError("Number of workers must be positive")
            
        if feature_type not in ["skeleton", "angles"]:
            raise ValueError(f"Invalid feature type. Must be one of: ['skeleton', 'angles']")
            
        if dance_style is not None and dance_style not in DANCE_STYLE_WEIGHTS:
            raise ValueError(f"Invalid dance style. Must be one of: {list(DANCE_STYLE_WEIGHTS.keys())}")
            
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.n_workers = n_workers
        self.dance_style = dance_style
        
        # Initialize metrics
        self.metrics = {
            'dtw': DTWMetric(n_workers=n_workers),
            'correlation': CorrelationMetric(n_workers=n_workers),
            'tlcc': TLCCMetric(n_workers=n_workers, dance_style=dance_style)
        }
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Validate dataset
        if not self.dataset.get('reference_sequences') or not self.dataset.get('query_sequences'):
            raise ValueError("No sequences found in dataset")
    
    def _load_dataset(self) -> Dict:
        """Load the dataset from the specified path."""
        dataset_file = os.path.join(self.dataset_path, f'{self.feature_type}_dataset.json')
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
            
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
            
        # Validate dataset structure
        if not dataset.get('reference_sequences') or not dataset.get('query_sequences'):
            raise ValueError("No sequences found in dataset")
            
        # Check for empty sequences
        for seq_type in ['reference_sequences', 'query_sequences']:
            for seq_name, seq_data in dataset[seq_type].items():
                if not seq_data.get('sequence'):
                    raise ValueError("No sequences found in dataset")
                    
        return dataset
    
    def _compute_metric_scores(self,
                             metric_name: str,
                             reference_seq: np.ndarray,
                             query_seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scores and confidence for a metric.
        
        Args:
            metric_name: Name of the metric to use
            reference_seq: Reference sequence
            query_seqs: List of query sequences
            
        Returns:
            Tuple of (scores, confidence)
        """
        metric = self.metrics[metric_name]
        
        if metric_name == 'tlcc':
            scores, lags = metric.compute_batch_similarity(reference_seq, query_seqs)
            scores = np.array(scores)  # Ensure scores is a numpy array
            # Use lag consistency as confidence measure
            if len(lags) > 1:
                confidence = 1.0 - (np.std(lags) / len(reference_seq))
            else:
                confidence = np.array([1.0])
        else:
            scores = metric.compute_batch_similarity(reference_seq, query_seqs)
            scores = np.array(scores)  # Ensure scores is a numpy array
            # Use score consistency as confidence measure
            confidence = np.abs(scores - 0.5) * 2  # Higher confidence for scores far from 0.5
            
        return scores, confidence
    
    def run_benchmark(self) -> pd.DataFrame:
        """
        Run benchmark on all metrics and return results.
        
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        # Process each reference sequence
        for ref_name, ref_data in tqdm(self.dataset['reference_sequences'].items(),
                                     desc="Processing reference sequences"):
            ref_seq = np.array(ref_data['sequence'])
            
            # Process each query sequence
            for query_name, query_data in self.dataset['query_sequences'].items():
                if 'label' not in query_data:
                    raise KeyError("Missing label in query sequence")
                    
                query_seq = np.array(query_data['sequence'])
                true_label = query_data['label']
                
                # Validate sequence shapes
                if query_seq.shape[0] != ref_seq.shape[0]:
                    raise ValueError("Inconsistent sequence lengths")
                if query_seq.shape[1] != ref_seq.shape[1]:
                    raise ValueError("Inconsistent number of joints")
                
                # Compute scores for each metric
                for metric_name in self.metrics:
                    scores, confidence = self._compute_metric_scores(
                        metric_name,
                        ref_seq,
                        [query_seq]  # Wrap in list for batch processing
                    )
                    
                    # Add results
                    results.append({
                        'reference': ref_name,
                        'query': query_name,
                        'metric': metric_name,
                        'score': float(scores[0]),  # Convert to float for JSON serialization
                        'confidence': float(confidence[0]),
                        'true_label': true_label
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_dir = os.path.join(self.dataset_path, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Save summary
        summary = df.groupby('metric').agg({
            'score': ['mean', 'std'],
            'confidence': ['mean', 'std']
        }).round(4)
        summary.to_csv(os.path.join(output_dir, 'summary.csv'))
        
        return df
    
    def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute metrics from benchmark results.
        
        Args:
            results: DataFrame with benchmark results
            
        Returns:
            DataFrame with metric scores
        """
        metrics_data = []
        
        # Group by metric and compute accuracy and confidence
        for metric_name in results['metric'].unique():
            metric_results = results[results['metric'] == metric_name]
            
            # Compute accuracy (correct predictions)
            correct = (metric_results['score'] > 0.5) == metric_results['true_label']
            accuracy = correct.mean()
            
            # Compute average confidence
            avg_confidence = metric_results['confidence'].mean()
            
            # Compute weighted accuracy
            weighted_accuracy = accuracy * avg_confidence
            
            metrics_data.append({
                'metric': metric_name,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'weighted_accuracy': weighted_accuracy
            })
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        df.set_index('metric', inplace=True)
        df = df[['accuracy', 'avg_confidence', 'weighted_accuracy']]  # Ensure column order
        return df
    
    def print_results(self, metrics_df: pd.DataFrame):
        """Print benchmark results in a formatted table."""
        print("\nBenchmark Results:")
        print("=" * 80)
        print(tabulate(metrics_df.sort_values('weighted_accuracy', ascending=False),
                      headers='keys',
                      tablefmt='grid',
                      floatfmt=".3f"))
        
        # Save results to file
        self.save_results(metrics_df)
    
    def save_results(self, metrics_df: pd.DataFrame) -> None:
        """Save benchmark results to a CSV file."""
        output_file = os.path.join(self.dataset_path, f"benchmark_results_{self.feature_type}.csv")
        metrics_df.to_csv(output_file)  # Save with index (metric column)
        print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run benchmark on dance pose metrics')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--feature_type', type=str, required=True,
                      choices=['skeleton', 'angles'],
                      help='Type of features to use')
    parser.add_argument('--dance_style', type=str,
                      choices=list(DANCE_STYLE_WEIGHTS.keys()),
                      help='Dance style for joint weighting')
    parser.add_argument('--n_workers', type=int, default=4,
                      help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Initialize and run benchmark
    runner = BenchmarkRunner(
        dataset_path=args.dataset_path,
        feature_type=args.feature_type,
        dance_style=args.dance_style,
        n_workers=args.n_workers
    )
    
    # Run benchmark and compute metrics
    results = runner.run_benchmark()
    metrics_df = runner.compute_metrics(results)
    
    # Print results
    runner.print_results(metrics_df)

if __name__ == '__main__':
    main() 