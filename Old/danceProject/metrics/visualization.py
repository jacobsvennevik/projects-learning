import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from .joint_weights import JOINT_INDICES, DANCE_STYLE_WEIGHTS

def plot_joint_importance(weights: Dict[int, float], 
                         title: str = "Joint Importance",
                         figsize: tuple = (12, 8),
                         cmap: str = "YlOrRd") -> plt.Figure:
    """
    Create a heatmap visualization of joint importance weights.
    
    Args:
        weights: Dictionary mapping joint indices to their weights
        title: Title for the plot
        figsize: Figure size (width, height)
        cmap: Color map for the heatmap
        
    Returns:
        Matplotlib figure object
    """
    # Create a matrix of weights
    weight_matrix = np.zeros((len(JOINT_INDICES), 1))
    for joint_idx, weight in weights.items():
        weight_matrix[joint_idx] = weight
        
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(weight_matrix, 
                cmap=cmap,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Weight'},
                ax=ax)
    
    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Joint Index')
    
    # Add joint names as y-tick labels
    joint_names = [name for name, _ in sorted(JOINT_INDICES.items(), key=lambda x: x[1])]
    ax.set_yticklabels(joint_names, rotation=0)
    
    plt.tight_layout()
    return fig

def plot_dance_style_comparison(styles: Optional[List[str]] = None,
                              figsize: tuple = (15, 10)) -> plt.Figure:
    """
    Create a comparison plot of joint importance across different dance styles.
    
    Args:
        styles: List of dance styles to compare. If None, uses all available styles.
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if styles is None:
        styles = list(DANCE_STYLE_WEIGHTS.keys())
        
    # Create a matrix of weights for all styles
    weight_matrix = np.zeros((len(JOINT_INDICES), len(styles)))
    for i, style in enumerate(styles):
        weights = DANCE_STYLE_WEIGHTS[style]
        for joint_idx, weight in weights.items():
            weight_matrix[joint_idx, i] = weight
            
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(weight_matrix,
                cmap="YlOrRd",
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Weight'},
                ax=ax)
    
    # Customize the plot
    ax.set_title("Joint Importance Across Dance Styles")
    ax.set_xlabel('Dance Style')
    ax.set_ylabel('Joint Index')
    
    # Add labels
    joint_names = [name for name, _ in sorted(JOINT_INDICES.items(), key=lambda x: x[1])]
    ax.set_yticklabels(joint_names, rotation=0)
    ax.set_xticklabels(styles, rotation=45)
    
    plt.tight_layout()
    return fig

def plot_joint_importance_radar(weights: Dict[int, float],
                              title: str = "Joint Importance Radar",
                              figsize: tuple = (10, 10)) -> plt.Figure:
    """
    Create a radar chart visualization of joint importance weights.
    
    Args:
        weights: Dictionary mapping joint indices to their weights
        title: Title for the plot
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Prepare data
    joint_names = []
    joint_weights = []
    for name, idx in sorted(JOINT_INDICES.items(), key=lambda x: x[1]):
        joint_names.append(name)
        joint_weights.append(weights.get(idx, 1.0))
    
    # Close the radar chart
    joint_names += [joint_names[0]]
    joint_weights += [joint_weights[0]]
    
    # Create angles
    angles = np.linspace(0, 2*np.pi, len(joint_names), endpoint=True)
    
    # Plot data
    ax.plot(angles, joint_weights)
    ax.fill(angles, joint_weights, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(joint_names[:-1])
    
    # Add title
    plt.title(title)
    
    plt.tight_layout()
    return fig

def plot_joint_importance_bar(weights: Dict[int, float],
                            title: str = "Joint Importance Bar Chart",
                            figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Create a bar chart visualization of joint importance weights.
    
    Args:
        weights: Dictionary mapping joint indices to their weights
        title: Title for the plot
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Prepare data
    joint_names = []
    joint_weights = []
    for name, idx in sorted(JOINT_INDICES.items(), key=lambda x: x[1]):
        joint_names.append(name)
        joint_weights.append(weights.get(idx, 1.0))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(joint_names, joint_weights)
    
    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel('Joint')
    ax.set_ylabel('Weight')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig 