import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

class ReportCardGenerator:
    """Generator for visualization report cards comparing different metrics."""
    
    def __init__(self, 
                 results_path: str,
                 output_dir: str,
                 feature_type: str,
                 dance_style: Optional[str] = None):
        """
        Initialize the report card generator.
        
        Args:
            results_path: Path to the benchmark results CSV
            output_dir: Directory to save visualizations
            feature_type: Type of features used
            dance_style: Optional dance style used
        """
        self.results_path = results_path
        self.output_dir = output_dir
        self.feature_type = feature_type
        self.dance_style = dance_style
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        self.results_df = pd.read_csv(results_path)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_accuracy_comparison(self, figsize: Tuple[int, int] = (10, 6)):
        """Create bar chart comparing accuracy across metrics."""
        plt.figure(figsize=figsize)
        
        metrics = self.results_df['metric'].unique()
        accuracies = [self.results_df[self.results_df['metric'] == m]['accuracy'].mean() 
                     for m in metrics]
        
        bars = plt.bar(metrics, accuracies)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.title('Accuracy Comparison Across Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_comparison.png'))
        plt.close()
    
    def plot_confidence_comparison(self, figsize: Tuple[int, int] = (10, 6)):
        """Create box plot comparing confidence distributions across metrics."""
        plt.figure(figsize=figsize)
        
        sns.boxplot(x='metric', y='confidence', data=self.results_df)
        
        plt.title('Confidence Distribution Across Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confidence_comparison.png'))
        plt.close()
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 5)):
        """Create confusion matrices for each metric."""
        n_metrics = len(self.results_df['metric'].unique())
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        for idx, metric in enumerate(self.results_df['metric'].unique()):
            metric_data = self.results_df[self.results_df['metric'] == metric]
            y_true = metric_data['true_label']
            y_pred = (metric_data['score'] > 0.5).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
            axes[idx].set_title(f'{metric.upper()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'))
        plt.close()
    
    def plot_roc_curves(self, figsize: Tuple[int, int] = (10, 8)):
        """Create ROC curves for each metric."""
        plt.figure(figsize=figsize)
        
        for metric in self.results_df['metric'].unique():
            metric_data = self.results_df[self.results_df['metric'] == metric]
            y_true = metric_data['true_label']
            y_score = metric_data['score']
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{metric.upper()} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Metrics')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'))
        plt.close()
    
    def plot_precision_recall_curves(self, figsize: Tuple[int, int] = (10, 8)):
        """Create precision-recall curves for each metric."""
        plt.figure(figsize=figsize)
        
        for metric in self.results_df['metric'].unique():
            metric_data = self.results_df[self.results_df['metric'] == metric]
            y_true = metric_data['true_label']
            y_score = metric_data['score']
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{metric.upper()} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Different Metrics')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curves.png'))
        plt.close()
    
    def plot_score_distributions(self, figsize: Tuple[int, int] = (12, 6)):
        """Create violin plots showing score distributions for each metric."""
        plt.figure(figsize=figsize)
        
        sns.violinplot(x='metric', y='score', data=self.results_df, hue='true_label')
        
        plt.title('Score Distributions by Metric and True Label')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_distributions.png'))
        plt.close()
    
    def generate_report_card(self):
        """Generate all visualizations for the report card."""
        print("Generating report card visualizations...")
        
        # Create all plots
        self.plot_accuracy_comparison()
        self.plot_confidence_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_score_distributions()
        
        # Generate summary statistics
        summary_stats = self.results_df.groupby('metric').agg({
            'accuracy': ['mean', 'std'],
            'confidence': ['mean', 'std'],
            'weighted_accuracy': ['mean', 'std']
        }).round(3)
        
        # Save summary statistics
        summary_stats.to_csv(os.path.join(self.output_dir, 'summary_statistics.csv'))
        
        print(f"Report card generated successfully in {self.output_dir}")
        print("Files generated:")
        print("- accuracy_comparison.png")
        print("- confidence_comparison.png")
        print("- confusion_matrices.png")
        print("- roc_curves.png")
        print("- precision_recall_curves.png")
        print("- score_distributions.png")
        print("- summary_statistics.csv")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization report card')
    parser.add_argument('--results_path', type=str, required=True,
                      help='Path to the benchmark results CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save visualizations')
    parser.add_argument('--feature_type', type=str, required=True,
                      help='Type of features used')
    parser.add_argument('--dance_style', type=str,
                      help='Dance style used (optional)')
    
    args = parser.parse_args()
    
    # Generate report card
    generator = ReportCardGenerator(
        results_path=args.results_path,
        output_dir=args.output_dir,
        feature_type=args.feature_type,
        dance_style=args.dance_style
    )
    
    generator.generate_report_card()

if __name__ == '__main__':
    main() 