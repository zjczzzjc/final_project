# Network Simulation Project

This project contains Python scripts for simulating and analyzing three classic network models:
- Erdos-Renyi (ER) Random Graph
- Watts-Strogatz (WS) Small-World Network
- Kleinberg Navigable Small-World Network

## Script Files

### 1. er_graph.py
**Purpose**: Erdos-Renyi Random Graph Model

**Features**:
- `er_graph_sample(n, p)` - Generates an ER random graph with n nodes and edge probability p
- `calc_degree_distribution()` - Computes and plots degree distribution
- `calc_clustering_coefficient()` - Computes clustering coefficient
- `calc_average_distance()` - Computes average shortest path length
- `calc_diameter()` - Computes network diameter
- `calc_component_sizes()` - Computes connected component sizes

**Usage**:
```bash
python3 er_graph.py
```
Then input n, lambda (average degree), and p (edge probability).

---

### 2. ws_graph.py
**Purpose**: Watts-Strogatz Small-World Network Model

**Features**:
- `ws_graph_sample(n, k, p)` - Generates a WS small-world graph with n nodes, average degree k, and rewiring probability p
- `calc_degree_distribution()` - Computes and plots degree distribution
- `calc_clustering_coefficient()` - Computes clustering coefficient
- `calc_average_distance()` - Computes average shortest path length
- `calc_diameter()` - Computes network diameter
- `calc_component_sizes()` - Computes connected component sizes
- `analyze_ws_p_effect()` - Analyzes how clustering coefficient and diameter change as p varies from 0 to 1

**Usage**:
```bash
python3 ws_graph.py
```
Analyzes the effect of rewiring probability p on network properties.

---

### 3. kleinberg_1d.py
**Purpose**: 1-Dimensional Kleinberg Navigable Small-World Network

**Features**:
- `kleinberg_1d_graph_sample(n, p_local, q_long, r)` - Generates a 1D Kleinberg graph with n nodes
- `calc_degree_distribution()` - Computes and plots degree distribution
- `calc_clustering_coefficient()` - Computes clustering coefficient
- `calc_average_distance()` - Computes average shortest path length
- `calc_diameter()` - Computes network diameter
- `calc_component_sizes()` - Computes connected component sizes
- `get_distance_distribution()` - Computes distance distribution between node pairs

**Parameters**:
- `p_local`: Maximum Hamming distance for local connections
- `q_long`: Number of long-range connections per node
- `r`: Decay exponent for long-range connection probability (P ∝ 1/d^r)

**Usage**:
```bash
python3 kleinberg_1d.py
```

---

### 4. kleinberg_d.py
**Purpose**: d-Dimensional Kleinberg Navigable Small-World Network

**Features**:
- `kleinberg_d_graph_sample(n, d, p_local, q_long, r)` - Generates a d-dimensional Kleinberg graph
- `greedy_routing_d()` - Implements greedy routing on the network
- `simulate_greedy_routing_d()` - Simulates greedy routing performance
- `compare_r_values()` - Compares greedy routing performance for different r values
- `compare_dimensions()` - Compares greedy routing performance across dimensions

**Usage**:
```bash
python3 kleinberg_d.py
```

---

### 5. compare_networks.py
**Purpose**: Comparison of ER, WS, and Kleinberg Network Models

**Features**:
- Compares three network models across multiple metrics:
  1. Clustering Coefficient
  2. Diameter
  3. Degree Distribution
  4. Component Size Distribution
  5. Distance Distribution

**Output**:
- Text summary of metrics
- Four comparison plots saved as PNG files:
  - `clustering_diameter_comparison.png`
  - `degree_distribution_comparison.png`
  - `component_size_distribution.png`
  - `distance_distribution.png`

**Usage**:
```bash
python3 compare_networks.py
```

---

### 6. visualize_routing.py
**Purpose**: Visualization of Greedy Routing on Kleinberg Networks

**Features**:
- Animates greedy routing on 1D and 2D Kleinberg networks
- Shows how messages navigate from source to target
- Creates GIF animations of routing paths

**Usage**:
```bash
python3 visualize_routing.py
```

---

## Required Dependencies

```bash
pip install matplotlib numpy
```

## Network Models Summary

| Model | Description | Key Properties |
|-------|-------------|----------------|
| **ER** | Random graph with independent edges | Low clustering, short average distance |
| **WS** | Regular lattice with random rewiring | High clustering + short average distance (small-world) |
| **Kleinberg** | Lattice with power-law long-range connections | Optimized for greedy routing (O(log²n) steps) |

## References

1. Erdos, P., & Renyi, A. (1959). On random graphs.
2. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of small-world networks.
3. Kleinberg, J. M. (2000). The small-world phenomenon: An algorithmic perspective.
