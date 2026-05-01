"""
3D spatial region for positioning agricultural markets (mandis).

Represents the geographic area where markets are located.
In the context of Indian agricultural markets:
- X, Y dimensions: Geographic coordinates (simplified)
- Z dimension: Elevation or market tier (could represent market size/importance)
"""

import numpy as np


class Region3D:
    """
    3D rectangular region for sampling market locations.
    
    Attributes:
        Lx: Length in X dimension (km or arbitrary units)
        Ly: Length in Y dimension (km or arbitrary units)  
        Lz: Length in Z dimension (elevation or market tier)
        rng: Random number generator
    """
    
    def __init__(self, Lx=100, Ly=100, Lz=100, seed=None):
        """
        Initialize a 3D rectangular region.
        
        Args:
            Lx: X dimension size (default 100)
            Ly: Y dimension size (default 100)
            Lz: Z dimension size (default 100)
            seed: Random seed for reproducibility
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.rng = np.random.default_rng(seed)
    
    def sample_nodes(self, n):
        """
        Sample n random positions uniformly in the region.
        
        Args:
            n: Number of positions to sample
        
        Returns:
            np.ndarray: Array of shape (n, 3) with random positions
        """
        x = self.rng.uniform(0, self.Lx, n)
        y = self.rng.uniform(0, self.Ly, n)
        z = self.rng.uniform(0, self.Lz, n)
        return np.column_stack((x, y, z))
    
    def sample_clustered_nodes(self, n, num_clusters=3, cluster_std=15):
        """
        Sample positions in clusters (realistic for regional market grouping).
        
        Markets often cluster around:
        - Agricultural production zones
        - Transportation hubs
        - Urban centers
        
        Args:
            n: Number of positions
            num_clusters: Number of market clusters
            cluster_std: Standard deviation of cluster spread
        
        Returns:
            np.ndarray: Array of shape (n, 3) with clustered positions
        """
        # Generate cluster centers
        centers = np.column_stack((
            self.rng.uniform(0.2 * self.Lx, 0.8 * self.Lx, num_clusters),
            self.rng.uniform(0.2 * self.Ly, 0.8 * self.Ly, num_clusters),
            self.rng.uniform(0.2 * self.Lz, 0.8 * self.Lz, num_clusters)
        ))
        
        # Assign each node to a cluster
        cluster_assignments = self.rng.choice(num_clusters, n)
        
        # Sample around cluster centers
        positions = []
        for i in range(n):
            center = centers[cluster_assignments[i]]
            offset = self.rng.normal(0, cluster_std, 3)
            pos = center + offset
            
            # Clip to region bounds
            pos[0] = np.clip(pos[0], 0, self.Lx)
            pos[1] = np.clip(pos[1], 0, self.Ly)
            pos[2] = np.clip(pos[2], 0, self.Lz)
            
            positions.append(pos)
        
        return np.array(positions)