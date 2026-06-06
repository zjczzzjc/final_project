import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def independent_bernoulli_sample(n, p):
    if p <= 0.0:
        return []
    if p >= 1.0:
        return list(range(1, n + 1))

    selected = []
    cur = 0
    log1_minus_p = math.log(1.0 - p)

    while True:
        u = random.random() 
        if u == 0.0:
            u = 1e-12
        x = math.floor(math.log(u) / log1_minus_p) + 1
        cur += x
        if cur > n:
            break
        selected.append(cur)

    return selected

def er_graph_sample(n, p):
    g = [[] for i in range(n + 1)]
    seq = independent_bernoulli_sample(n * (n - 1) // 2, p)

    for i in seq:
        u = int(math.ceil((math.sqrt(8 * i + 1) + 1) / 2))
        v = i - (u - 1) * (u - 2) // 2
        assert(not (u is v))
        g[u].append(v)
        g[v].append(u)
    return g

def calc_degree_distribution(n, p, g):
    lambda_val = p * n
    deg = [len(g[i]) for i in range(n + 1)]
    m = int(math.log(n))
    cnt = [0 for i in range(m + 1)]
    for i in range(1, n + 1):
        if deg[i] <= m:
            cnt[deg[i]] += 1

    plt.figure(f'Experiment for (n, lambda) = ({n}, {p * n})')
    x_axis = np.arange(m + 1)
    width = 0.35
    poisson = [math.exp(-lambda_val) * math.pow(lambda_val, i) / math.factorial(i) for i in range(m + 1)]
    plt.bar(x_axis - width / 2, [x / n for x in cnt], width, label = 'Degree distribution', color = 'red')
    plt.bar(x_axis + width / 2, poisson, width, label = 'Poisson distribution', color = 'blue')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(f'er_degree_dist_n{n}_lambda{lambda_val:.2f}.png', dpi=300, bbox_inches='tight')

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

def analyze_er_network(n, p):
    g = er_graph_sample(n, p)
    
    cc = calc_clustering_coefficient(n, g)
    avg_dist = calc_average_distance(n, g)
    diameter = calc_diameter(n, g)
    component_sizes = calc_component_sizes(n, g)
    
    print(f'Erdos-Renyi Model (n={n}, p={p})')
    print(f'Clustering Coefficient: {cc:.4f}')
    print(f'Average Distance: {avg_dist:.4f}')
    print(f'Diameter: {diameter}')
    print(f'Largest Component Size: {component_sizes[0] if component_sizes else 0}')
    print()
    
    return g, cc, avg_dist, diameter

if __name__ == '__main__':
    print('Input the number of vertices:')
    n = int(input())
    print('Input the parameter lambda: ')
    p = float(input()) / n
    g = er_graph_sample(n, p)

    calc_degree_distribution(n, p, g)
    plt.show()

    cc = calc_clustering_coefficient(n, g)
    avg_dist = calc_average_distance(n, g)
    diameter = calc_diameter(n, g)
    
    print(f'Clustering Coefficient: {cc:.4f}')
    print(f'Average Distance: {avg_dist:.4f}')
    print(f'Diameter: {diameter}')