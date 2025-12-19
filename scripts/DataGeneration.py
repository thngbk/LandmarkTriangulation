import numpy as np
from sklearn.datasets import make_blobs
import argparse

def generate_synthetic_data(n_samples=5000, n_features=50, n_clusters=5):
    """
    Generates high-dimensional synthetic data with distinct clusters.
    """
    print(f"Generating {n_samples} samples with {n_features} dimensions and {n_clusters} clusters...")
    
    # Generate isotropic Gaussian blobs for clustering
    X, y = make_blobs(n_samples=n_samples, 
                      n_features=n_features, 
                      centers=n_clusters, 
                      cluster_std=2.0, # Spread of clusters
                      random_state=42)
    
    return X, y

