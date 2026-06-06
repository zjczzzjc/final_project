import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from er_graph import er_graph_sample
from ws_graph import ws_graph_sample
from kleinberg_1d import kleinberg_1d_graph_sample

def bfs_distances(g, n, start):
    dist = [-1] * (n + 1)
    q = deque()
    q.append(start)
    dist[start] = 0
    while q:
        u = q.popleft()
        for v in g[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def get_distance_distribution(g, n):
    all_distances = []
    for i in range(1, min(n, 100)):
        dist = bfs_distances(g, n, i)
        for j in range(i + 1, n + 1):
            if dist[j] != -1:
                all_distances.append(dist[j])
    return all_distances

def get_degree_dist(g, n):
    deg = [len(g[i]) for i in range(1, n + 1)]
    max_deg = max(deg)
    cnt = [0] * (max_deg + 1)
    for d in deg:
        cnt[d] += 1
    return cnt, max_deg

def get_distance_counts(distances):
    if not distances:
        return [], 0
    max_dist = max(distances)
    counts = [0] * (max_dist + 1)
    for d in distances:
        counts[d] += 1
    return counts, max_dist

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

def plot_clustering_and_diameter(er_cc, ws_cc, kb_cc, er_diameter, ws_diameter, kb_diameter):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bar_width = 0.25
    index = np.arange(1)
    
    ax1.bar(index - bar_width, [er_cc], bar_width, label='Erdos-Renyi', color='red')
    ax1.bar(index, [ws_cc], bar_width, label='Watts-Strogatz', color='green')
    ax1.bar(index + bar_width, [kb_cc], bar_width, label='Kleinberg', color='orange')
    ax1.set_xticks(index)
    ax1.set_xticklabels(['Clustering Coefficient'])
    ax1.set_ylabel('Value')
    ax1.set_title('Clustering Coefficient Comparison')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.bar(index - bar_width, [er_diameter], bar_width, label='Erdos-Renyi', color='red')
    ax2.bar(index, [ws_diameter], bar_width, label='Watts-Strogatz', color='green')
    ax2.bar(index + bar_width, [kb_diameter], bar_width, label='Kleinberg', color='orange')
    ax2.set_xticks(index)
    ax2.set_xticklabels(['Diameter'])
    ax2.set_ylabel('Value')
    ax2.set_title('Diameter Comparison')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('clustering_diameter_comparison.png', dpi=300, bbox_inches='tight')
    print('Figure saved as: clustering_diameter_comparison.png')
    plt.show()

def plot_degree_distribution(er_g, ws_g, kb_g, n):
    er_deg_cnt, er_deg_max = get_degree_dist(er_g, n)
    ws_deg_cnt, ws_deg_max = get_degree_dist(ws_g, n)
    kb_deg_cnt, kb_deg_max = get_degree_dist(kb_g, n)
    max_degree = max(er_deg_max, ws_deg_max, kb_deg_max)
    x = np.arange(max_degree + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - 0.24, [er_deg_cnt[d] / n if d <= er_deg_max else 0 for d in x], width=0.24, label='Erdos-Renyi', color='red', alpha=0.7)
    ax.bar(x, [ws_deg_cnt[d] / n if d <= ws_deg_max else 0 for d in x], width=0.24, label='Watts-Strogatz', color='green', alpha=0.7)
    ax.bar(x + 0.24, [kb_deg_cnt[d] / n if d <= kb_deg_max else 0 for d in x], width=0.24, label='Kleinberg', color='orange', alpha=0.7)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Probability')
    ax.set_title('Degree Distribution Comparison')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('degree_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print('Figure saved as: degree_distribution_comparison.png')
    plt.show()

def plot_component_size_distribution(er_components_sorted, ws_components_sorted, kb_components_sorted):
    max_comp_size = max(len(er_components_sorted), len(ws_components_sorted), len(kb_components_sorted))
    x_comp = np.arange(max_comp_size)
    er_comp_padded = er_components_sorted + [0] * (max_comp_size - len(er_components_sorted))
    ws_comp_padded = ws_components_sorted + [0] * (max_comp_size - len(ws_components_sorted))
    kb_comp_padded = kb_components_sorted + [0] * (max_comp_size - len(kb_components_sorted))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x_comp - 0.24, er_comp_padded, width=0.24, label='Erdos-Renyi', color='red', alpha=0.7)
    ax.bar(x_comp, ws_comp_padded, width=0.24, label='Watts-Strogatz', color='green', alpha=0.7)
    ax.bar(x_comp + 0.24, kb_comp_padded, width=0.24, label='Kleinberg', color='orange', alpha=0.7)
    ax.set_xlabel('Component Rank')
    ax.set_ylabel('Component Size')
    ax.set_title('Component Size Distribution (Sorted Descending)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlim(-0.5, min(20, max_comp_size - 0.5))
    
    plt.tight_layout()
    plt.savefig('component_size_distribution.png', dpi=300, bbox_inches='tight')
    print('Figure saved as: component_size_distribution.png')
    plt.show()

def plot_distance_distribution(er_g, ws_g, kb_g, n):
    er_distances = get_distance_distribution(er_g, n)
    ws_distances = get_distance_distribution(ws_g, n)
    kb_distances = get_distance_distribution(kb_g, n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if er_distances and ws_distances and kb_distances:
        er_counts, er_max = get_distance_counts(er_distances)
        ws_counts, ws_max = get_distance_counts(ws_distances)
        kb_counts, kb_max = get_distance_counts(kb_distances)
        
        max_dist = max(er_max, ws_max, kb_max)
        
        er_probs = [er_counts[d] / len(er_distances) if d <= er_max else 0 for d in range(max_dist + 1)]
        ws_probs = [ws_counts[d] / len(ws_distances) if d <= ws_max else 0 for d in range(max_dist + 1)]
        kb_probs = [kb_counts[d] / len(kb_distances) if d <= kb_max else 0 for d in range(max_dist + 1)]
        
        bar_width = 0.24
        x = np.arange(max_dist + 1)
        
        ax.bar(x - bar_width, er_probs, width=bar_width, label='Erdos-Renyi', color='red', alpha=0.7)
        ax.bar(x, ws_probs, width=bar_width, label='Watts-Strogatz', color='green', alpha=0.7)
        ax.bar(x + bar_width, kb_probs, width=bar_width, label='Kleinberg', color='orange', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(i)) for i in x])
    
    ax.set_xlabel('Distance')
    ax.set_ylabel('Probability')
    ax.set_title('Distance Distribution')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('distance_distribution.png', dpi=300, bbox_inches='tight')
    print('Figure saved as: distance_distribution.png')
    plt.show()

def compare_networks(n=1000, er_lambda=10, ws_k=10, ws_p=0.1, kb_p_local=1, kb_q_long=2, kb_r=1):
    er_p = er_lambda / n
    
    er_g = er_graph_sample(n, er_p)
    ws_g = ws_graph_sample(n, ws_k, ws_p)
    kb_g, _, _ = kleinberg_1d_graph_sample(n, kb_p_local, kb_q_long, kb_r)
    
    er_cc = calc_clustering_coefficient(n, er_g)
    er_diameter = calc_diameter(n, er_g)
    er_components = calc_component_sizes(n, er_g)
    er_components_sorted = sorted(er_components, reverse=True)
    
    ws_cc = calc_clustering_coefficient(n, ws_g)
    ws_diameter = calc_diameter(n, ws_g)
    ws_components = calc_component_sizes(n, ws_g)
    ws_components_sorted = sorted(ws_components, reverse=True)
    
    kb_cc = calc_clustering_coefficient(n, kb_g)
    kb_diameter = calc_diameter(n, kb_g)
    kb_components = calc_component_sizes(n, kb_g)
    kb_components_sorted = sorted(kb_components, reverse=True)
    
    print('=' * 70)
    print('Network Model Comparison Results (ER vs WS vs Kleinberg)')
    print('=' * 70)
    print(f'Parameters: n={n}, ER(lambda={er_lambda}), WS(k={ws_k}, p={ws_p}), Kleinberg(p={kb_p_local}, q={kb_q_long}, r={kb_r})')
    print('=' * 70)
    print(f'{"Metric":<25} {"Erdos-Renyi":<15} {"Watts-Strogatz":<15} {"Kleinberg":<15}')
    print('-' * 70)
    print(f'{"Clustering Coefficient":<25} {er_cc:<15.4f} {ws_cc:<15.4f} {kb_cc:<15.4f}')
    print(f'{"Diameter":<25} {er_diameter:<15} {ws_diameter:<15} {kb_diameter:<15}')
    print(f'{"Largest Component Size":<25} {er_components_sorted[0]:<15} {ws_components_sorted[0]:<15} {kb_components_sorted[0]:<15}')
    print('=' * 70)
    
    plot_clustering_and_diameter(er_cc, ws_cc, kb_cc, er_diameter, ws_diameter, kb_diameter)
    plot_degree_distribution(er_g, ws_g, kb_g, n)
    plot_component_size_distribution(er_components_sorted, ws_components_sorted, kb_components_sorted)
    plot_distance_distribution(er_g, ws_g, kb_g, n)

if __name__ == '__main__':
    print('ER vs WS vs Kleinberg Network Model Comparison Tool')
    print('=' * 50)
    
    print('\n1. Comparison of ER, WS, and Kleinberg Models')
    compare_networks(n=1000, er_lambda=10, ws_k=10, ws_p=0.2, kb_p_local=4, kb_q_long=2, kb_r=1)
    
    print('\nComparison complete!')
