import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def ws_graph_sample(n, k, p):
    g = [[] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j - 1) % n + 1
            g[i].append(neighbor)
            g[neighbor].append(i)
    
    for i in range(1, n + 1):
        old_neighbors = g[i].copy()
        for old_neighbor in old_neighbors:
            if old_neighbor > i:
                if random.random() < p:
                    new_neighbor = random.randint(1, n)
                    while new_neighbor == i or new_neighbor in g[i]:
                        new_neighbor = random.randint(1, n)
                    g[i].remove(old_neighbor)
                    g[old_neighbor].remove(i)
                    g[i].append(new_neighbor)
                    g[new_neighbor].append(i)
    
    return g

def calc_degree_distribution(n, g):
    deg = [len(g[i]) for i in range(n + 1)]
    max_deg = max(deg)
    cnt = [0] * (max_deg + 1)
    for i in range(1, n + 1):
        cnt[deg[i]] += 1
    
    plt.figure(f'WS Degree Distribution (n={n})')
    x_axis = np.arange(max_deg + 1)
    plt.bar(x_axis, [x / n for x in cnt], width=0.8, label='Degree distribution', color='green')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Watts-Strogatz Degree Distribution')
    plt.legend()
    plt.savefig(f'ws_degree_dist_n{n}.png', dpi=300, bbox_inches='tight')

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
            for k in range(j + 1, len(neighbors)):
                if neighbors[k] in g[neighbors[j]]:
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
    
    return total_distance / pairs_count if pairs_count > 0 else 0.0

def calc_diameter(n, g):
    max_dist = 0
    
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
        
        current_max = max(d for d in dist if d != -1)
        max_dist = max(max_dist, current_max)
    
    return max_dist

def calc_component_sizes(n, g):
    vis = [False] * (n + 1)
    sizes = []
    
    for i in range(1, n + 1):
        if not vis[i]:
            que = deque([i])
            vis[i] = True
            size = 0
            
            while que:
                u = que.popleft()
                size += 1
                for v in g[u]:
                    if not vis[v]:
                        vis[v] = True
                        que.append(v)
            
            sizes.append(size)
    
    sizes.sort(reverse=True)
    return sizes

def analyze_ws_network(n, k, p):
    g = ws_graph_sample(n, k, p)
    
    cc = calc_clustering_coefficient(n, g)
    avg_dist = calc_average_distance(n, g)
    diameter = calc_diameter(n, g)
    component_sizes = calc_component_sizes(n, g)
    
    print(f'Watts-Strogatz Model (n={n}, k={k}, p={p})')
    print(f'Clustering Coefficient: {cc:.4f}')
    print(f'Average Distance: {avg_dist:.4f}')
    print(f'Diameter: {diameter}')
    print(f'Largest Component Size: {component_sizes[0] if component_sizes else 0}')
    print()
    
    return g, cc, avg_dist, diameter

def analyze_ws_p_effect(n, k, num_trials=5):
    p_values = np.linspace(0, 1, 21)
    
    cc_values = []
    diameter_values = []
    
    print('=' * 70)
    print(f'WS Model - Effect of Rewiring Probability p')
    print(f'Parameters: n={n}, k={k}, trials={num_trials}')
    print('=' * 70)
    print(f'{"p":<10} {"Avg CC":<15} {"Avg Diameter":<15}')
    print('-' * 70)
    
    for p in p_values:
        cc_sum = 0
        diameter_sum = 0
        
        for _ in range(num_trials):
            g = ws_graph_sample(n, k, p)
            cc = calc_clustering_coefficient(n, g)
            diameter = calc_diameter(n, g)
            cc_sum += cc
            diameter_sum += diameter
        
        avg_cc = cc_sum / num_trials
        avg_diameter = diameter_sum / num_trials
        
        cc_values.append(avg_cc)
        diameter_values.append(avg_diameter)
        
        print(f'{p:<10.2f} {avg_cc:<15.4f} {avg_diameter:<15}')
    
    print('=' * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(p_values, cc_values, marker='o', color='green', linewidth=2, markersize=6, label='Simulation')
    
    cc_theoretical = [3 * (k - 2) * ((1 - p) ** 3) / (4 * (k - 1)) for p in p_values]
    ax1.plot(p_values, cc_theoretical, marker='', color='red', linewidth=2, linestyle='--', label='Theoretical')
    
    ax1.set_xlabel('Rewiring Probability p')
    ax1.set_ylabel('Clustering Coefficient')
    ax1.set_title(f'Clustering Coefficient vs. Rewiring Probability (n={n}, k={k})')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, 1)
    ax1.legend()
    
    ax2.plot(p_values, diameter_values, marker='s', color='blue', linewidth=2, markersize=6, label='Simulation')
    
    diameter_theoretical = []
    for p in p_values:
        x = p * k * n / 2
        if x == 0:
            f_x = 0
        else:
            sqrt_term = np.sqrt(x ** 2 + 4 * x)
            f_x = (4 / sqrt_term) * np.arctanh(x / sqrt_term)
        l_p = (n / k) * f_x
        diameter_theoretical.append(l_p)
    
    ax2.plot(p_values, diameter_theoretical, marker='', color='red', linewidth=2, linestyle='--', label='Theoretical')
    
    ax2.set_xlabel('Rewiring Probability p')
    ax2.set_ylabel('Diameter')
    ax2.set_title(f'Diameter vs. Rewiring Probability (n={n}, k={k})')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('ws_p_effect.png', dpi=300, bbox_inches='tight')
    print('Figure saved as: ws_p_effect.png')
    plt.show()

if __name__ == '__main__':
    print('Watts-Strogatz Model Analysis')
    print('=' * 50)
    
    print('\n1. Effect of Rewiring Probability p')
    analyze_ws_p_effect(n=200, k=10, num_trials=5)
    
    print('\nAnalysis complete!')
