import time
import numpy as np
from grid import Grid
from heuristics import HEURISTICS, manhattan
from astar import astar
from prioritized_planning import independent_astar, prioritized_planning
from cbs import cbs
from hybrid_pp_cbs import hybrid_pp_cbs


def run_q1_benchmarks(grid_sizes=None, obstacle_densities=None, num_trials=50):
    """
    Run Q1 benchmarks: compare heuristics for single-agent A*.
    
    Returns:
        dict: results[grid_size][density][heuristic_name] = {
            'nodes_expanded': list, 'time': list, 'cost': list, 'nodes_generated': list
        }
    """
    if grid_sizes is None:
        grid_sizes = [20, 30, 50]
    if obstacle_densities is None:
        obstacle_densities = [0.1, 0.2, 0.3]
    
    results = {}
    
    for size in grid_sizes:
        results[size] = {}
        for density in obstacle_densities:
            results[size][density] = {}
            for h_name in HEURISTICS:
                results[size][density][h_name] = {
                    'nodes_expanded': [],
                    'nodes_generated': [],
                    'time': [],
                    'cost': []
                }
            
            print(f"  Running {size}x{size}, density={density}...")
            
            for trial in range(num_trials):
                # Use same grid for all heuristics (fair comparison)
                seed = trial * 1000 + size * 100 + int(density * 10)
                grid = Grid(size, size, density, seed=seed)
                
                for h_name, h_func in HEURISTICS.items():
                    result = astar(grid, grid.start, grid.goal, h_func)
                    
                    if result['path'] is not None:
                        results[size][density][h_name]['nodes_expanded'].append(result['nodes_expanded'])
                        results[size][density][h_name]['nodes_generated'].append(result['nodes_generated'])
                        results[size][density][h_name]['time'].append(result['time'])
                        results[size][density][h_name]['cost'].append(result['cost'])
    
    return results


def run_q2_benchmarks(grid_sizes=None, agent_counts=None, obstacle_densities=None, num_trials=30):
    """
    Run Q2 benchmarks: compare Plain A* vs Prioritized Planning vs CBS.
    
    Returns:
        dict: results[grid_size][num_agents][density][method] = {
            'sum_of_costs': list, 'makespan': list, 'collisions': list,
            'time': list, 'success_count': int, 'total_count': int
        }
    """
    if grid_sizes is None:
        grid_sizes = [10, 15, 20]
    if agent_counts is None:
        agent_counts = [2, 3, 5, 8]
    if obstacle_densities is None:
        obstacle_densities = [0.1, 0.2]
    
    methods = ['Independent A*', 'Prioritized Planning', 'CBS', 'H-PCBS']
    results = {}
    
    for size in grid_sizes:
        results[size] = {}
        for num_agents in agent_counts:
            results[size][num_agents] = {}
            for density in obstacle_densities:
                results[size][num_agents][density] = {}
                for method in methods:
                    results[size][num_agents][density][method] = {
                        'sum_of_costs': [],
                        'makespan': [],
                        'collisions': [],
                        'time': [],
                        'success_count': 0,
                        'total_count': 0
                    }
                
                print(f"  Running {size}x{size}, agents={num_agents}, density={density}...")
                
                for trial in range(num_trials):
                    seed = trial * 10000 + size * 100 + num_agents * 10 + int(density * 10)
                    
                    try:
                        grid, starts, goals = Grid.generate_multi_agent(
                            size, size, num_agents, density, seed=seed
                        )
                    except (ValueError, Exception):
                        continue
                    
                    # Method 1: Independent A*
                    try:
                        r = independent_astar(grid, starts, goals, manhattan)
                        d = results[size][num_agents][density]['Independent A*']
                        d['total_count'] += 1
                        if r['success']:
                            d['success_count'] += 1
                            d['sum_of_costs'].append(r['sum_of_costs'])
                            d['makespan'].append(r['makespan'])
                            d['collisions'].append(r['collisions'])
                            d['time'].append(r['time'])
                    except Exception:
                        pass
                    
                    # Method 2: Prioritized Planning
                    try:
                        r = prioritized_planning(grid, starts, goals, manhattan)
                        d = results[size][num_agents][density]['Prioritized Planning']
                        d['total_count'] += 1
                        if r['success']:
                            d['success_count'] += 1
                            d['sum_of_costs'].append(r['sum_of_costs'])
                            d['makespan'].append(r['makespan'])
                            d['collisions'].append(r['collisions'])
                            d['time'].append(r['time'])
                    except Exception:
                        pass
                    
                    # Method 3: CBS
                    try:
                        r = cbs(grid, starts, goals, manhattan, time_limit=5)
                        d = results[size][num_agents][density]['CBS']
                        d['total_count'] += 1
                        if r['success']:
                            d['success_count'] += 1
                            d['sum_of_costs'].append(r['sum_of_costs'])
                            d['makespan'].append(r['makespan'])
                            d['collisions'].append(0)  # CBS guarantees 0 collisions
                            d['time'].append(r['time'])
                    except Exception:
                        pass
    
                    # Method 4: Hybrid Prioritized Planning + CBS
                    try:
                        r = hybrid_pp_cbs(
                            grid, starts, goals, manhattan, time_limit=5
                        )
                        d = results[size][num_agents][density]['H-PCBS']
                        d['total_count'] += 1

                        if r['success']:
                            d['success_count'] += 1
                            d['sum_of_costs'].append(r['sum_of_costs'])
                            d['makespan'].append(r['makespan'])
                            d['collisions'].append(r['collisions'])
                            d['time'].append(r['time'])
                    except Exception:
                        pass

    return results


def summarize_q1(results):
    """Print summary table for Q1 results."""
    print("\n" + "=" * 80)
    print("Q1 RESULTS: Single-Agent A* Heuristic Comparison")
    print("=" * 80)
    
    for size in sorted(results.keys()):
        for density in sorted(results[size].keys()):
            print(f"\nGrid: {size}x{size}, Obstacle Density: {density}")
            print(f"{'Heuristic':<20} {'Nodes Expanded':>15} {'Time (ms)':>12} {'Path Cost':>12}")
            print("-" * 60)
            for h_name in results[size][density]:
                data = results[size][density][h_name]
                if data['nodes_expanded']:
                    avg_nodes = np.mean(data['nodes_expanded'])
                    avg_time = np.mean(data['time']) * 1000
                    avg_cost = np.mean(data['cost'])
                    print(f"{h_name:<20} {avg_nodes:>15.1f} {avg_time:>12.3f} {avg_cost:>12.1f}")


def summarize_q2(results):
    """Print summary table for Q2 results."""
    print("\n" + "=" * 80)
    print("Q2 RESULTS: Multi-Agent Pathfinding Comparison")
    print("=" * 80)
    
    for size in sorted(results.keys()):
        for num_agents in sorted(results[size].keys()):
            for density in sorted(results[size][num_agents].keys()):
                print(f"\nGrid: {size}x{size}, Agents: {num_agents}, Density: {density}")
                print(f"{'Method':<25} {'Success%':>10} {'Collisions':>12} {'SOC':>10} {'Makespan':>10} {'Time(ms)':>10}")
                print("-" * 80)
                for method in results[size][num_agents][density]:
                    d = results[size][num_agents][density][method]
                    if d['total_count'] > 0:
                        success_pct = (d['success_count'] / d['total_count']) * 100
                        avg_col = np.mean(d['collisions']) if d['collisions'] else 0
                        avg_soc = np.mean(d['sum_of_costs']) if d['sum_of_costs'] else 0
                        avg_ms = np.mean(d['makespan']) if d['makespan'] else 0
                        avg_time = np.mean(d['time']) * 1000 if d['time'] else 0
                        print(f"{method:<25} {success_pct:>9.1f}% {avg_col:>12.1f} {avg_soc:>10.1f} {avg_ms:>10.1f} {avg_time:>10.2f}")
