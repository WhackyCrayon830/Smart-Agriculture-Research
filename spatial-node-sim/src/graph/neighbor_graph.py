"""
Enhanced graph builder for agricultural market (mandi) networks.

Creates realistic market network topologies based on:
- Geographic proximity (k-NN)
- Transportation infrastructure
- Trade volume potential
- Regional market hierarchies
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(nodes, k=3, distance_weighted=True):
    """
    Build k-nearest neighbor graph for mandi network.
    
    Creates connections between geographically close markets, which is
    realistic for agricultural trade where:
    - Transportation costs are distance-dependent
    - Information flows between nearby markets
    - Arbitrage opportunities exist within transport range
    
    Args:
        nodes: List of Node objects with position attribute
        k: Number of neighbors per node (default 3)
        distance_weighted: If True, store distances for weighted interactions
    
    Returns:
        dict: Adjacency dictionary {node_id: [neighbor_ids]}
              If distance_weighted, also updates node.neighbor_distances
    """
    positions = np.array([n.position for n in nodes])
    n_nodes = len(positions)
    
    # Build k-NN index
    # Use k+1 because each node is its own nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n_nodes)).fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    
    graph = {}
    
    for i, (neigh_idxs, neigh_dists) in enumerate(zip(indices, distances)):
        # Skip first neighbor (the node itself, distance=0)
        neighbor_ids = [int(idx) for idx in neigh_idxs[1:]]
        neighbor_dists = neigh_dists[1:]
        
        graph[i] = neighbor_ids
        
        # Store distances in node for weighted calculations
        if distance_weighted and hasattr(nodes[i], 'neighbor_distances'):
            nodes[i].neighbor_distances = neighbor_dists.tolist()
    
    return graph


def build_hierarchical_graph(nodes, k=3, hub_fraction=0.2):
    """
    Build hierarchical market network with hub-spoke structure.
    
    Realistic for Indian agricultural markets where:
    - Major mandis act as regional hubs (higher connectivity)
    - Smaller mandis connect primarily to nearby hubs
    - Some direct connections between similar-tier markets
    
    Args:
        nodes: List of Node objects
        k: Base number of neighbors for regular nodes
        hub_fraction: Fraction of nodes designated as hubs
    
    Returns:
        dict: Hierarchical adjacency dictionary
    """
    n_nodes = len(nodes)
    positions = np.array([n.position for n in nodes])
    
    # Identify hubs (randomly for now, could be based on position centrality)
    n_hubs = max(1, int(hub_fraction * n_nodes))
    hub_indices = np.random.choice(n_nodes, n_hubs, replace=False)
    hub_set = set(hub_indices)
    
    graph = {}
    
    # For each node
    for i in range(n_nodes):
        if i in hub_set:
            # Hubs connect to many neighbors (2x base connectivity)
            k_local = k * 2
        else:
            k_local = k
        
        # Find k nearest neighbors
        dists = np.linalg.norm(positions - positions[i], axis=1)
        # Exclude self
        dists[i] = np.inf
        
        # Prefer connecting to hubs if not a hub yourself
        if i not in hub_set:
            # Boost hub connection priority
            for hub_idx in hub_indices:
                if hub_idx != i:
                    dists[hub_idx] *= 0.5  # Make hubs "closer"
        
        # Get k nearest
        nearest = np.argsort(dists)[:k_local]
        graph[i] = nearest.tolist()
    
    return graph


def build_transport_network(nodes, k=3, road_quality_variance=0.3):
    """
    Build graph based on transportation infrastructure simulation.
    
    Models realistic transport networks where:
    - Major highways reduce effective distance
    - Poor roads increase effective distance
    - Some connections are better than others
    
    Args:
        nodes: List of Node objects
        k: Number of neighbors
        road_quality_variance: Variance in road quality (0-1)
    
    Returns:
        dict: Transport-weighted adjacency dictionary
    """
    positions = np.array([n.position for n in nodes])
    n_nodes = len(positions)
    
    # Generate random road quality matrix (symmetric)
    road_quality = np.random.uniform(
        1.0 - road_quality_variance,
        1.0 + road_quality_variance,
        (n_nodes, n_nodes)
    )
    # Make symmetric
    road_quality = (road_quality + road_quality.T) / 2
    
    graph = {}
    
    for i in range(n_nodes):
        # Calculate effective distances (physical * road_quality)
        physical_dists = np.linalg.norm(positions - positions[i], axis=1)
        effective_dists = physical_dists * road_quality[i]
        effective_dists[i] = np.inf  # Exclude self
        
        # Connect to k nearest by effective distance
        nearest = np.argsort(effective_dists)[:k]
        graph[i] = nearest.tolist()
        
        # Store effective distances for transport cost calculations
        if hasattr(nodes[i], 'neighbor_distances'):
            nodes[i].neighbor_distances = effective_dists[nearest].tolist()
    
    return graph


def add_long_distance_links(graph, nodes, num_links=None, max_distance=None):
    """
    Add long-distance trading links to simulate major trade routes.
    
    In real markets, some distant mandis have strong trade relationships
    despite geographic distance, due to:
    - Specialized crops
    - Established trade relationships
    - Major transportation corridors
    
    Args:
        graph: Existing adjacency dictionary (modified in-place)
        nodes: List of Node objects
        num_links: Number of long-distance links to add (default: 10% of nodes)
        max_distance: Maximum distance for long links (None = no limit)
    
    Returns:
        dict: Updated graph with long-distance links
    """
    n_nodes = len(nodes)
    if num_links is None:
        num_links = max(1, n_nodes // 10)
    
    positions = np.array([n.position for n in nodes])
    
    for _ in range(num_links):
        # Pick two random nodes
        i, j = np.random.choice(n_nodes, 2, replace=False)
        
        # Check distance constraint
        if max_distance is not None:
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist > max_distance:
                continue
        
        # Add bidirectional link if not already present
        if j not in graph[i]:
            graph[i].append(j)
        if i not in graph[j]:
            graph[j].append(i)
    
    return graph


def analyze_network(graph, nodes):
    """
    Analyze network topology and return statistics.
    
    Useful for understanding:
    - Network connectivity
    - Hub identification
    - Degree distribution
    - Average path length
    
    Args:
        graph: Adjacency dictionary
        nodes: List of Node objects
    
    Returns:
        dict: Network statistics
    """
    degrees = [len(neighbors) for neighbors in graph.values()]
    
    # Calculate some basic stats
    stats = {
        'num_nodes': len(nodes),
        'num_edges': sum(degrees) // 2,  # Undirected edges
        'avg_degree': np.mean(degrees),
        'min_degree': np.min(degrees),
        'max_degree': np.max(degrees),
        'degree_std': np.std(degrees),
        'density': sum(degrees) / (len(nodes) * (len(nodes) - 1))
    }
    
    # Identify hubs (nodes with degree > mean + std)
    hub_threshold = stats['avg_degree'] + stats['degree_std']
    hubs = [i for i, deg in enumerate(degrees) if deg > hub_threshold]
    stats['num_hubs'] = len(hubs)
    stats['hub_nodes'] = hubs
    
    return stats


# Default graph builder (can be swapped for different topologies)
def build_default_mandi_network(nodes, k=3, add_long_links=True):
    """
    Build default mandi network with realistic topology.
    
    Combines:
    - k-NN for local connectivity
    - Optional long-distance links for major trade routes
    
    Args:
        nodes: List of Node objects
        k: Number of nearest neighbors
        add_long_links: Whether to add long-distance trading links
    
    Returns:
        dict: Complete adjacency dictionary
    """
    # Base k-NN graph
    graph = build_knn_graph(nodes, k=k, distance_weighted=True)
    
    # Add some long-distance links (10% of nodes)
    if add_long_links and len(nodes) > 10:
        graph = add_long_distance_links(graph, nodes)
    
    return graph