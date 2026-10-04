import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from grid import Grid

# Color scheme
COLORS = {
    'obstacle': '#2c3e50',
    'free': '#ecf0f1',
    'start': '#27ae60',
    'goal': '#e74c3c',
    'path': '#3498db',
    'grid_line': '#bdc3c7'
}

AGENT_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']


def plot_grid(grid, path=None, title="Grid", filename=None, show=False):
    """
    Plot a grid with obstacles, start, goal, and optional path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw grid cells
    for x in range(grid.width):
        for y in range(grid.height):
            if (x, y) in grid.obstacles:
                color = COLORS['obstacle']
            else:
                color = COLORS['free']
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=COLORS['grid_line'], linewidth=0.5)
            ax.add_patch(rect)
    
    # Draw path
    if path:
        for i, (px, py) in enumerate(path):
            if (px, py) != grid.start and (px, py) != grid.goal:
                rect = plt.Rectangle((px, py), 1, 1, facecolor=COLORS['path'], alpha=0.6,
                                   edgecolor=COLORS['grid_line'], linewidth=0.5)
                ax.add_patch(rect)
        # Draw path line
        path_x = [p[0] + 0.5 for p in path]
        path_y = [p[1] + 0.5 for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
    
    # Draw start and goal
    start_rect = plt.Rectangle((grid.start[0], grid.start[1]), 1, 1,
                                facecolor=COLORS['start'], edgecolor='black', linewidth=2)
    goal_rect = plt.Rectangle((grid.goal[0], grid.goal[1]), 1, 1,
                               facecolor=COLORS['goal'], edgecolor='black', linewidth=2)
    ax.add_patch(start_rect)
    ax.add_patch(goal_rect)
    
    ax.text(grid.start[0] + 0.5, grid.start[1] + 0.5, 'S', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(grid.goal[0] + 0.5, grid.goal[1] + 0.5, 'G', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
    if show:
        plt.show()
    plt.close()


def plot_multi_agent(grid, paths, starts, goals, title="Multi-Agent Paths", filename=None, show=False):
    """
    Plot multiple agent paths on a single grid.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw grid
    for x in range(grid.width):
        for y in range(grid.height):
            if (x, y) in grid.obstacles:
                color = COLORS['obstacle']
            else:
                color = COLORS['free']
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=COLORS['grid_line'], linewidth=0.5)
            ax.add_patch(rect)
    
    # Draw each agent's path
    for i, path in enumerate(paths):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        if path and len(path) > 1:
            path_x = [p[0] + 0.5 for p in path]
            path_y = [p[1] + 0.5 for p in path]
            ax.plot(path_x, path_y, '-', color=color, linewidth=2.5, alpha=0.7, label=f'Agent {i}')
        
        # Draw start
        ax.add_patch(plt.Rectangle((starts[i][0], starts[i][1]), 1, 1,
                                    facecolor=color, alpha=0.8, edgecolor='black', linewidth=2))
        ax.text(starts[i][0] + 0.5, starts[i][1] + 0.5, f'S{i}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        
        # Draw goal
        ax.add_patch(plt.Rectangle((goals[i][0], goals[i][1]), 1, 1,
                                    facecolor=color, alpha=0.4, edgecolor='black', linewidth=2))
        ax.text(goals[i][0] + 0.5, goals[i][1] + 0.5, f'G{i}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='black')
    
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.invert_yaxis()
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
    if show:
        plt.show()
    plt.close()


def plot_q1_results(results, output_dir='.'):
    """
    Generate comparison plots for Q1 (single-agent heuristic comparison).
    """
    import os
    
    # Plot 1: Nodes Expanded comparison (bar chart for each grid size)
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, size in enumerate(sorted(results.keys())):
        ax = axes[idx]
        # Use middle density
        densities = sorted(results[size].keys())
        density = densities[len(densities) // 2]  # middle density
        
        heuristic_names = list(results[size][density].keys())
        means = []
        stds = []
        for h_name in heuristic_names:
            data = results[size][density][h_name]['nodes_expanded']
            means.append(np.mean(data) if data else 0)
            stds.append(np.std(data) if data else 0)
        
        bars = ax.bar(range(len(heuristic_names)), means, yerr=stds,
                     capsize=5, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
                     edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(heuristic_names)))
        ax.set_xticklabels(heuristic_names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Nodes Expanded')
        ax.set_title(f'{size}x{size} Grid (density={density})')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Q1: Nodes Expanded by Heuristic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_nodes_expanded.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    
    # Plot 2: Execution Time comparison
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, size in enumerate(sorted(results.keys())):
        ax = axes[idx]
        densities = sorted(results[size].keys())
        density = densities[len(densities) // 2]
        
        heuristic_names = list(results[size][density].keys())
        means = [np.mean(results[size][density][h]['time']) * 1000
                 if results[size][density][h]['time'] else 0
                 for h in heuristic_names]
        stds = [np.std(results[size][density][h]['time']) * 1000
                if results[size][density][h]['time'] else 0
                for h in heuristic_names]
        
        ax.bar(range(len(heuristic_names)), means, yerr=stds,
              capsize=5, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
              edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(heuristic_names)))
        ax.set_xticklabels(heuristic_names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{size}x{size} Grid (density={density})')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Q1: Execution Time by Heuristic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_execution_time.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    
    # Plot 3: Path Cost verification (all should be equal)
    fig, ax = plt.subplots(figsize=(10, 5))
    size = sorted(results.keys())[len(results) // 2]  # middle size
    density = sorted(results[size].keys())[len(results[size]) // 2]
    
    for h_name in results[size][density]:
        costs = results[size][density][h_name]['cost']
        if costs:
            ax.plot(costs, 'o-', label=h_name, alpha=0.7, markersize=3)
    
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Path Cost')
    ax.set_title(f'Q1: Path Cost Verification ({size}x{size}, density={density})\nAll admissible heuristics yield identical optimal cost', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_path_cost_verification.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    
    # Plot 4: Nodes expanded across different obstacle densities
    fig, ax = plt.subplots(figsize=(10, 6))
    size = sorted(results.keys())[len(results) // 2]
    
    x_labels = []
    for density in sorted(results[size].keys()):
        x_labels.append(f"{density}")
    
    x = np.arange(len(x_labels))
    width = 0.2
    heuristic_names = list(results[size][sorted(results[size].keys())[0]].keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, h_name in enumerate(heuristic_names):
        means = []
        for density in sorted(results[size].keys()):
            data = results[size][density][h_name]['nodes_expanded']
            means.append(np.mean(data) if data else 0)
        ax.bar(x + i * width, means, width, label=h_name, color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Obstacle Density')
    ax.set_ylabel('Average Nodes Expanded')
    ax.set_title(f'Q1: Impact of Obstacle Density on Nodes Expanded ({size}x{size})')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_density_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def plot_q2_results(results, output_dir='.'):
    """
    Generate comparison plots for Q2 (multi-agent comparison).
    """
    import os
    
    # Aggregate across densities for cleaner plots
    # Plot 1: Collision Count vs Number of Agents
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Independent A*', 'Prioritized Planning', 'CBS', 'H-PCBS']
    method_colors = {'Independent A*': '#e74c3c', 'Prioritized Planning': '#3498db', 'CBS': '#2ecc71', 'H-PCBS': '#f39c12'}
    
    size = sorted(results.keys())[len(results) // 2]  # middle grid size
    agent_counts = sorted(results[size].keys())
    
    for method in methods:
        avg_collisions = []
        for num_agents in agent_counts:
            all_collisions = []
            for density in results[size][num_agents]:
                d = results[size][num_agents][density][method]
                if d['collisions']:
                    all_collisions.extend(d['collisions'])
            avg_collisions.append(np.mean(all_collisions) if all_collisions else 0)
        
        ax.plot(agent_counts, avg_collisions, 'o-', color=method_colors[method],
               label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Average Collisions', fontsize=12)
    ax.set_title(f'Q2: Collision Count vs Number of Agents ({size}x{size} Grid)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q2_collisions.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    
    # Plot 2: Sum of Costs vs Number of Agents
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        avg_soc = []
        for num_agents in agent_counts:
            all_soc = []
            for density in results[size][num_agents]:
                d = results[size][num_agents][density][method]
                if d['sum_of_costs']:
                    all_soc.extend(d['sum_of_costs'])
            avg_soc.append(np.mean(all_soc) if all_soc else 0)
        
        ax.plot(agent_counts, avg_soc, 'o-', color=method_colors[method],
               label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Sum of Costs', fontsize=12)
    ax.set_title(f'Q2: Sum of Costs vs Number of Agents ({size}x{size} Grid)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q2_sum_of_costs.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    
    # Plot 3: Runtime comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        avg_time = []
        for num_agents in agent_counts:
            all_times = []
            for density in results[size][num_agents]:
                d = results[size][num_agents][density][method]
                if d['time']:
                    all_times.extend(d['time'])
            avg_time.append(np.mean(all_times) * 1000 if all_times else 0)
        
        ax.plot(agent_counts, avg_time, 'o-', color=method_colors[method],
               label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
    ax.set_title(f'Q2: Runtime vs Number of Agents ({size}x{size} Grid)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q2_runtime.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    
    # Plot 4: Success Rate vs Number of Agents
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        success_rates = []
        for num_agents in agent_counts:
            total_success = 0
            total_trials = 0
            for density in results[size][num_agents]:
                d = results[size][num_agents][density][method]
                total_success += d['success_count']
                total_trials += d['total_count']
            rate = (total_success / total_trials * 100) if total_trials > 0 else 0
            success_rates.append(rate)
        
        ax.plot(agent_counts, success_rates, 'o-', color=method_colors[method],
               label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Q2: Success Rate vs Number of Agents ({size}x{size} Grid)', fontsize=14)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q2_success_rate.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
    
    # Plot 5: Makespan comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        avg_makespan = []
        for num_agents in agent_counts:
            all_ms = []
            for density in results[size][num_agents]:
                d = results[size][num_agents][density][method]
                if d['makespan']:
                    all_ms.extend(d['makespan'])
            avg_makespan.append(np.mean(all_ms) if all_ms else 0)
        
        ax.plot(agent_counts, avg_makespan, 'o-', color=method_colors[method],
               label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Makespan (time steps)', fontsize=12)
    ax.set_title(f'Q2: Makespan vs Number of Agents ({size}x{size} Grid)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q2_makespan.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()
