import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def torus_distance(a, b, grid_size):
    return min(abs(a - b), grid_size - abs(a - b))

def kleinberg_d_graph_sample(n, d, p_local, q_long, r=2):
    grid_size = int(n ** (1.0 / d))
    if grid_size ** d < n:
        grid_size += 1
    
    pos = {}
    idx = 1
    
    def generate_coords(dim, current):
        nonlocal idx
        if dim == d:
            if idx <= n:
                pos[idx] = tuple(current)
                idx += 1
            return
        
        for i in range(grid_size):
            current.append(i)
            generate_coords(dim + 1, current)
            current.pop()
    
    generate_coords(0, [])
    
    g = [[] for _ in range(n + 1)]
    
    for u in range(1, n + 1):
        weights = []
        targets = []
        
        for v in range(1, n + 1):
            if u == v:
                continue
            
            hamming_dist = sum(torus_distance(a, b, grid_size) for a, b in zip(pos[u], pos[v]))
            
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

def greedy_routing_d(g, pos, grid_size, d, start, target):
    if start == target:
        return 0, True
    
    visited = set()
    current = start
    steps = 0
    
    while current != target:
        if current in visited:
            unvisited_neighbors = [n for n in g[current] if n not in visited]
            if not unvisited_neighbors:
                return steps, False
            
            best_neighbor = None
            best_distance = float('inf')
            for neighbor in unvisited_neighbors:
                hamming_dist = sum(torus_distance(a, b, grid_size) for a, b in zip(pos[neighbor], pos[target]))
                if hamming_dist < best_distance:
                    best_distance = hamming_dist
                    best_neighbor = neighbor
            current = best_neighbor
            steps += 1
            continue
        
        visited.add(current)
        
        best_neighbor = None
        best_distance = float('inf')
        
        for neighbor in g[current]:
            hamming_dist = sum(torus_distance(a, b, grid_size) for a, b in zip(pos[neighbor], pos[target]))
            
            if hamming_dist < best_distance:
                best_distance = hamming_dist
                best_neighbor = neighbor
        
        if best_neighbor is None:
            return steps, False
        
        current = best_neighbor
        steps += 1
        
        if steps > grid_size ** d * 10:
            return steps, False
    
    return steps, True

def simulate_greedy_routing_d(n, d, p_local, q_long, r=2, num_trials=3):
    total_steps = 0
    success_count = 0
    
    for _ in range(num_trials):
        g, pos, grid_size = kleinberg_d_graph_sample(n, d, p_local, q_long, r)
        
        pos_to_node = {v: k for k, v in pos.items()}
        
        start_pos = tuple([0] * d)
        target_pos = tuple([grid_size // 2] * d)
        
        start = pos_to_node.get(start_pos, 1)
        target = pos_to_node.get(target_pos, math.ceil(n / 2))
        
        steps, success = greedy_routing_d(g, pos, grid_size, d, start, target)
        if success:
            total_steps += steps
            success_count += 1
    
    avg_steps = total_steps / success_count if success_count > 0 else float('inf')
    success_rate = success_count / num_trials
    
    return avg_steps, success_rate

def compare_r_values(n=1000, d=2, p_local=1, q_long=2, num_trials=3):
    #r_values = [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    r_values = np.linspace(d, d + 5, 11)
    
    avg_steps_list = []
    success_rates = []
    
    m = int(n ** (1.0 / d) / 2)
    
    print('=' * 70)
    print(f'{d}-D Kleinberg Model - Greedy Routing vs. Decay Exponent r')
    print(f'Parameters: n={n}, d={d}, m={m}, p_local={p_local}, q_long={q_long}, trials={num_trials}')
    print('=' * 70)
    print(f'{"r":<10} {"Avg Steps":<15}')
    print('-' * 70)
    
    for r in r_values:
        avg_steps, success_rate = simulate_greedy_routing_d(n, d, p_local, q_long, r, num_trials)
        avg_steps_list.append(avg_steps)
        success_rates.append(success_rate)
        print(f'{r:<10.1f} {avg_steps:<15.2f}')
    
    print('=' * 70)
    
    theoretical_steps = []
    for r in r_values:
        if r < d - 1e-9:
            exponent = (d - r) / (d + 1)
            steps = m ** exponent
        elif abs(r - d) < 1e-9:
            steps = (math.log(m)) ** 2
        else:
            exponent = (r - d) / (r - d + 1)
            steps = m ** exponent
        steps = steps / (q_long * (p_local ** d))
        theoretical_steps.append(steps)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(r_values, avg_steps_list, marker='o', color='red', linewidth=2, label='Simulation')
    ax.plot(r_values, theoretical_steps, marker='^', color='green', linewidth=2, linestyle='--', label='Theoretical')
    
    ax.axvline(x=d, color='blue', linestyle=':', label=f'Optimum r={d}')
    
    ax.set_xlabel('Decay Exponent r')
    ax.set_ylabel('Average Routing Steps')
    ax.set_title(f'{d}-D Kleinberg Model - Greedy Routing Steps vs. r (m={m})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f'kleinberg_d{d}_routing_vs_r.png', dpi=300, bbox_inches='tight')
    print(f'Figure saved as: kleinberg_d{d}_routing_vs_r.png')
    plt.show()

def compare_dimensions(n=1000, p_local=1, q_long=2, r=2, num_trials=3):
    dimensions = [1, 2, 3]
    results = []
    
    print('=' * 70)
    print(f'Kleinberg Model - Dimension Comparison')
    print(f'Parameters: n={n}, p_local={p_local}, q_long={q_long}, r={r}, trials={num_trials}')
    print('=' * 70)
    print(f'{"Dimension":<10} {"Avg Steps":<15} ')
    print('-' * 70)
    
    for d in dimensions:
        avg_steps, success_rate = simulate_greedy_routing_d(n, d, p_local, q_long, r, num_trials)
        results.append((d, avg_steps, success_rate))
        print(f'{d:<10} {avg_steps:<15.2f} {success_rate:<15.2%}')
    
    print('=' * 70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([str(d) + 'D' for d, _, _ in results], [res[1] for res in results], color=['red', 'green', 'blue'])
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Average Routing Steps')
    ax.set_title('Greedy Routing Steps vs. Network Dimension')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('kleinberg_dimension_comparison.png', dpi=300, bbox_inches='tight')
    print('Figure saved as: kleinberg_dimension_comparison.png')
    plt.show()

if __name__ == '__main__':
    print('d-Dimensional Kleinberg Model')
    print('=' * 50)
    
    compare_r_values(n=100, d=2, p_local=1, q_long=1, num_trials=50)
    
    print('\nComparison complete!')