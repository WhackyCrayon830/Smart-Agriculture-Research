from src.region.region3d import Region3D
from src.simulation.node import Node
from src.graph.neighbor_graph import build_knn_graph
from src.field.weather_field_3d import WeatherField3D

# Create region
region = Region3D(100, 100, 100, seed=42)

# Sample coordinates
coords = region.sample_nodes(5)

# Create nodes
nodes = [Node(i, tuple(pos)) for i, pos in enumerate(coords)]

# Build neighbor graph
graph = build_knn_graph(nodes, k=2)

for node_id, neigh_list in graph.items():
    nodes[node_id].neighbors = neigh_list

# External field
field = WeatherField3D(seed=42)

# Simulation parameters
alpha = 0.6   # self memory
beta = 0.3    # field influence
gamma = 0.2   # neighbor influence

# Time simulation
for t in range(10):

    new_states = []

    for n in nodes:
        field_val = field.value(n.position, t)

        # Neighbor influence (average)
        if n.neighbors:
            neigh_states = [nodes[j].state for j in n.neighbors]
            neighbor_avg = sum(neigh_states) / len(neigh_states)
        else:
            neighbor_avg = 0

        new_state = (
            alpha * n.state +
            beta * field_val +
            gamma * (neighbor_avg - n.state)
        )

        new_states.append(new_state)

    # Apply updates after computing all
    for i, n in enumerate(nodes):
        n.state = new_states[i]

    # Print timestep
    print(f"\nTime {t}")
    for n in nodes:
        print("Node", n.id, "State:", round(n.state,4))
