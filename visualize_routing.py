import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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
                    if u_rand < cumsum :
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

def greedy_routing_trace(g, pos, grid_size, d, start, target):
    path = [start]
    visited = set([start])
    current = start
    
    while current != target:
        best_neighbor = None
        best_distance = float('inf')
        
        for neighbor in g[current]:
            distance = sum(torus_distance(a, b, grid_size) for a, b in zip(pos[neighbor], pos[target]))
            
            if distance < best_distance:
                best_distance = distance
                best_neighbor = neighbor
        
        if best_neighbor is None or best_neighbor in visited:
            return path, False
        
        current = best_neighbor
        path.append(current)
        visited.add(current)
        
        if len(path) > grid_size ** d * 10:
            return path, False
    
    return path, True

def visualize_routing_1d(g, pos, grid_size, path, start, target, r):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title('1D Greedy Routing on Kleinberg Torus (Circular Layout)')
    
    radius = 0.8
    
    def idx_to_angle(idx):
        return 2 * math.pi * idx / grid_size
    
    def idx_to_pos(idx):
        angle = idx_to_angle(idx)
        return (radius * math.cos(angle), radius * math.sin(angle))
    
    nodes = []
    for i in range(grid_size):
        x, y = idx_to_pos(i)
        nodes.append((x, y))
        ax.scatter(x, y, s=100, c='blue', alpha=0.6, zorder=2)
    
    for i in range(grid_size):
        x1, y1 = idx_to_pos(i)
        x2, y2 = idx_to_pos((i + 1) % grid_size)
        ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=2)
    
    long_edges = []
    for u in range(1, len(g)):
        for v in g[u]:
            if u < v:
                dist = torus_distance(pos[u][0], pos[v][0], grid_size)
                if dist > 1:
                    long_edges.append((pos[u][0], pos[v][0]))
    
    for idx1, idx2 in long_edges:
        x1, y1 = idx_to_pos(idx1)
        x2, y2 = idx_to_pos(idx2)
        
        ax.plot([x1, x2], [y1, y2], color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
    
    start_idx = pos[start][0]
    target_idx = pos[target][0]
    
    start_x, start_y = idx_to_pos(start_idx)
    target_x, target_y = idx_to_pos(target_idx)
    
    ax.scatter(start_x, start_y, s=200, c='green', marker='*', zorder=5)
    ax.scatter(target_x, target_y, s=200, c='red', marker='*', zorder=5)
    
    current_node, = ax.plot([], [], 'yo', markersize=15, zorder=4)
    path_line, = ax.plot([], [], 'r-', linewidth=2, zorder=3)
    
    def init():
        current_node.set_data([], [])
        path_line.set_data([], [])
        return current_node, path_line
    
    def update(frame):
        if frame < len(path):
            current = path[frame]
            current_idx = pos[current][0]
            x, y = idx_to_pos(current_idx)
            current_node.set_data([x], [y])
            
            if frame > 0:
                path_x = []
                path_y = []
                for i in range(frame + 1):
                    node = path[i]
                    idx = pos[node][0]
                    x_i, y_i = idx_to_pos(idx)
                    path_x.append(x_i)
                    path_y.append(y_i)
                path_line.set_data(path_x, path_y)
        
        return current_node, path_line
    
    anim = animation.FuncAnimation(fig, update, frames=len(path) + 5, 
                                   init_func=init, interval=800, blit=True, repeat=False)
    
    filename = f'greedy_routing_1d_{grid_size}_{r}.gif'
    writer = animation.PillowWriter(fps=2)
    anim.save(filename, writer=writer)
    print(f'Animation saved as: {filename}')
    
    plt.show()

def visualize_routing_2d(g, pos, grid_size, path, start, target, r):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title('2D Greedy Routing on Kleinberg Torus')
                                   
    all_edges = []
    long_edges = []
    
    for u in range(1, len(g)):
        for v in g[u]:
            if u < v:
                dx = torus_distance(pos[u][0], pos[v][0], grid_size)
                dy = torus_distance(pos[u][1], pos[v][1], grid_size)
                if dx + dy == 1:
                    all_edges.append((pos[u], pos[v]))
                else:
                    long_edges.append((pos[u], pos[v]))

    for edge in all_edges:
        x1, y1 = edge[0]
        x2, y2 = edge[1]
        
        if abs(x1 - x2) + abs(y1 - y2) == 1:
            ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=1, alpha=0.3)
        else:
            ax.plot([x1, x2], [y1, y2], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    for i in range(grid_size):
        ax.plot([i, i], [0, grid_size - 1], color='gray', linestyle='-', linewidth=1, alpha=0.3)
    
    for i in range(grid_size):
        ax.plot([0, grid_size - 1], [i, i], color='gray', linestyle='-', linewidth=1, alpha=0.3)
    
    ax.plot([0, grid_size - 1], [0, grid_size - 1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.plot([0, grid_size - 1], [grid_size - 1, 0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    for edge in long_edges:
        x1, y1 = edge[0]
        x2, y2 = edge[1]
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        if dx <= grid_size / 2 and dy <= grid_size / 2:
            ax.plot([x1, x2], [y1, y2], color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
        else:
            ax.plot([x1, x2], [y1, y2], color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    
    nodes = ax.scatter([pos[i][0] for i in range(1, len(g))], 
                       [pos[i][1] for i in range(1, len(g))], 
                       s=100, c='blue', alpha=0.6, zorder=2)
    
    start_node = ax.scatter(pos[start][0], pos[start][1], s=200, c='green', marker='*', zorder=5)
    target_node = ax.scatter(pos[target][0], pos[target][1], s=200, c='red', marker='*', zorder=5)
    
    current_node, = ax.plot([], [], 'yo', markersize=15, zorder=4)
    path_line, = ax.plot([], [], 'r-', linewidth=2, zorder=3)
    
    def init():
        current_node.set_data([], [])
        path_line.set_data([], [])
        return current_node, path_line
    
    def update(frame):
        if frame < len(path):
            current = path[frame]
            current_node.set_data([pos[current][0]], [pos[current][1]])
            
            if frame > 0:
                path_x = [pos[path[i]][0] for i in range(frame + 1)]
                path_y = [pos[path[i]][1] for i in range(frame + 1)]
                path_line.set_data(path_x, path_y)
        
        return current_node, path_line
    
    anim = animation.FuncAnimation(fig, update, frames=len(path) + 5, 
                                   init_func=init, interval=800, blit=True, repeat=False)
    
    filename = f'greedy_routing_2d_{grid_size}_{r}.gif'
    writer = animation.PillowWriter(fps=2)
    anim.save(filename, writer=writer)
    print(f'Animation saved as: {filename}')
    
    plt.show()

def visualize_routing_video(n=100, d=2, p_local=1, q_long=2, r=2):
    g, pos, grid_size = kleinberg_d_graph_sample(n, d, p_local, q_long, r)
    
    pos_to_node = {v: k for k, v in pos.items()}
    
    start_pos = tuple([0] * d)
    target_pos = tuple([grid_size // 2] * d)
    
    start = pos_to_node.get(start_pos, 1)
    target = pos_to_node.get(target_pos, math.ceil(n / 2))
    
    print(f'Dimension: {d}')
    print(f'Grid size: {grid_size}')
    print(f'Start position: {start_pos}, Node ID: {start}')
    print(f'Target position: {target_pos}, Node ID: {target}')
    
    path, success = greedy_routing_trace(g, pos, grid_size, d, start, target)
    
    if d == 1:
        visualize_routing_1d(g, pos, grid_size, path, start, target, r)
    elif d == 2:
        visualize_routing_2d(g, pos, grid_size, path, start, target, r)
    else:
        print(f'Visualization for d={d} not supported')
    
    print(f'Path length: {len(path) - 1} steps')
    print(f'Success: {success}')

if __name__ == '__main__':
    print('Visualizing Greedy Routing on Kleinberg Model')
    print('=' * 50)
    
    #print('\n1. 1D Kleinberg Model:')
    #visualize_routing_video(n=100, d=1, p_local=1, q_long=1, r=0)
    
    #print('\n2. 2D Kleinberg Model:')
    visualize_routing_video(n=400, d=2, p_local=1, q_long=1, r=10)
    
    print('\nAnimation complete!')