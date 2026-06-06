import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def torus_distance(a, b, grid_size):
    return min(abs(a - b), grid_size - abs(a - b))

def kleinberg_1d_graph_sample(n, p_local, q_long, r=1):
    grid_size = n
    
    pos = {}
    for i in range(1, n + 1):
        pos[i] = (i - 1,)
    
    g = [[] for _ in range(n + 1)]
    
    for u in range(1, n + 1):
        weights = []
        targets = []
        
        for v in range(1, n + 1):
            if u == v:
                continue
            
            hamming_dist = torus_distance(pos[u][0], pos[v][0], grid_size)
            
            if hamming_dist <= p_local:
                if v > u:
                    g[u].append(v)
                    g[v].append(u)
            elif hamming_dist > p_local:
                weight = 1.0 / (hamming_dist ** r)
                weights.append(weight)
                targets.append(v)
        
        if weights:
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            num_long_range = int(q_long)
            
            selected = []
            remaining_targets = targets.copy()
            remaining_probs = probs.copy()
            
            for _ in range(num_long_range):
                if not remaining_targets:
                    break
                
                u_rand = random.random()
                cumsum = 0
                selected_idx = -1
                
                for i, prob in enumerate(remaining_probs):
                    cumsum += prob
                    if u_rand < cumsum:
                        selected_idx = i
                        break
                
                if selected_idx >= 0:
                    target = remaining_targets.pop(selected_idx)
                    remaining_probs.pop(selected_idx)
                    
                    if remaining_probs:
                        remaining_probs = [p / sum(remaining_probs) for p in remaining_probs]
                    
                    if target not in g[u]:
                        selected.append(target)
            
            for v in selected:
                g[u].append(v)
                g[v].append(u)
    
    return g, pos, grid_size

def calc_degree_distribution(n, g, r=1, p_local=1, q_long=2):
    deg = [len(g[i]) for i in range(1, n + 1)]
    max_deg = max(deg)
    cnt = [0] * (max_deg + 1)
    for d in deg:
        cnt[d] += 1
    
    plt.figure(f'1D Kleinberg Degree Distribution (r={r})')
    x_axis = np.arange(max_deg + 1)
    plt.bar(x_axis, [x / n for x in cnt], color='orange')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Degree Distribution (n={n}, r={r}, p_local={p_local}, q_long={q_long})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'kleinberg_1d_degree_dist_n{n}_r{r}.png', dpi=300, bbox_inches='tight')
    return cnt, max_deg

def calc_clustering_coefficient(n, g):
    total_cc = 0.0
    count = 0
    
    for i in range(1, n + 1):
        neighbors = g[i]
        k_i = len(neighbors)
        if k_i < 2:
            continue
        
        possible_edges = k_i * (k_i - 1) // 2
        actual_edges = 0
        
        for j in range(len(neighbors)):
            for k_idx in range(j + 1, len(neighbors)):
                if neighbors[k_idx] in g[neighbors[j]]:
                    actual_edges += 1
        
        if possible_edges > 0:
            total_cc += actual_edges / possible_edges
            count += 1
    
    return total_cc / count if count > 0 else 0.0

def calc_average_distance(n, g):
    total_distance = 0
    pairs_count = 0
    
    for start in range(1, n + 1):
        dist = [-1] * (n + 1)
        que = deque([start])
        dist[start] = 0
        
        while que:
            u = que.popleft()
            for v in g[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    que.append(v)
        
        for end in range(start + 1, n + 1):
            if dist[end] != -1:
                total_distance += dist[end]
                pairs_count += 1
    
    return total_distance / pairs_count if pairs_count > 0 else float('inf')

def calc_diameter(n, g):
    max_distance = 0
    
    for start in range(1, n + 1):
        dist = [-1] * (n + 1)
        que = deque([start])
        dist[start] = 0
        
        while que:
            u = que.popleft()
            for v in g[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    que.append(v)
        
        for end in range(1, n + 1):
            if dist[end] > max_distance:
                max_distance = dist[end]
    
    return max_distance

def calc_component_sizes(n, g):
    visited = [False] * (n + 1)
    component_sizes = []
    
    for i in range(1, n + 1):
        if not visited[i]:
            que = deque([i])
            visited[i] = True
            size = 0
            
            while que:
                u = que.popleft()
                size += 1
                for v in g[u]:
                    if not visited[v]:
                        visited[v] = True
                        que.append(v)
            
            component_sizes.append(size)
    
    component_sizes.sort(reverse=True)
    return component_sizes

def get_distance_distribution(n, g):
    all_distances = []
    for i in range(1, min(n, 100)):
        dist = [-1] * (n + 1)
        que = deque([i])
        dist[i] = 0
        
        while que:
            u = que.popleft()
            for v in g[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    que.append(v)
        
        for j in range(i + 1, n + 1):
            if dist[j] != -1:
                all_distances.append(dist[j])
    return all_distances

if __name__ == '__main__':
    print('1D Kleinberg Model')
    print('=' * 50)
    
    n = 1000
    p_local = 1
    q_long = 2
    r = 1.0
    
    g, pos, grid_size = kleinberg_1d_graph_sample(n, p_local, q_long, r)
    
    print(f'n={n}, p_local={p_local}, q_long={q_long}, r={r}')
    
    cc = calc_clustering_coefficient(n, g)
    diameter = calc_diameter(n, g)
    components = calc_component_sizes(n, g)
    
    print(f'Clustering Coefficient: {cc:.4f}')
    print(f'Diameter: {diameter}')
    print(f'Largest Component: {components[0]}')
    
    calc_degree_distribution(n, g, r, p_local, q_long)
    plt.show()
