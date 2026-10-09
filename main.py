import sys
import os
from grid import Grid
from heuristics import HEURISTICS, manhattan
from astar import astar
from prioritized_planning import independent_astar, prioritized_planning
from cbs import cbs
from benchmarks import run_q1_benchmarks, run_q2_benchmarks, summarize_q1, summarize_q2
from visualize import plot_grid, plot_multi_agent, plot_q1_results, plot_q2_results


def run_demo(multi_agent_seed=128):
    """Run a visual demo of single and multi-agent pathfinding."""
    print("\n" + "=" * 60)
    print("DEMO: Single-Agent A* Pathfinding")
    print("=" * 60)
    
    # Single-agent demo
    grid = Grid(15, 15, obstacle_density=0.2, seed=42)
    print(f"Grid: {grid}")
    print(f"Start: {grid.start}, Goal: {grid.goal}")
    
    for h_name, h_func in HEURISTICS.items():
        result = astar(grid, grid.start, grid.goal, h_func)
        if result['path']:
            print(f"  {h_name}: path_cost={result['cost']}, "
                  f"nodes_expanded={result['nodes_expanded']}, "
                  f"time={result['time']*1000:.2f}ms")
    
    # Save demo grid
    result = astar(grid, grid.start, grid.goal, manhattan)
    plot_grid(grid, result['path'], title="Single-Agent A* (Manhattan Heuristic)",
             filename='demo_single_agent.png')
    
    print("\n" + "=" * 60)
    print("DEMO: Multi-Agent Pathfinding")
    print("=" * 60)
    
    # Multi-agent demo
    grid, starts, goals = Grid.generate_multi_agent(15, 15, 3, obstacle_density=0.15, seed=multi_agent_seed)
    print(f"Grid: {grid}")
    for i in range(len(starts)):
        print(f"  Agent {i}: {starts[i]} -> {goals[i]}")
    
    # Independent A*
    result_ind = independent_astar(grid, starts, goals)
    col_list = result_ind.get('collision_list', [])
    details = []
    e_cnt = sum(1 for c in col_list if c['type'] == 'edge')
    v_cnt = sum(1 for c in col_list if c['type'] == 'vertex')
    if e_cnt:
        details.append(f"{e_cnt} edge swap")
    if v_cnt:
        details.append(f"{v_cnt} vertex")
    col_str = f"{result_ind['collisions']}"
    if details:
        col_str += f" ({', '.join(details)})"
    print(f"\nIndependent A*:")
    print(f"  Sum of Costs: {result_ind['sum_of_costs']}, Collisions: {col_str}")
    plot_multi_agent(grid, result_ind['paths'], starts, goals,
                    title=f"Independent A* (Collisions: {col_str})",
                    filename='demo_independent_astar.png')
    
    # Prioritized Planning
    result_pp = prioritized_planning(grid, starts, goals)
    print(f"\nPrioritized Planning:")
    print(f"  Sum of Costs: {result_pp['sum_of_costs']}, Collisions: {result_pp['collisions']}")
    plot_multi_agent(grid, result_pp['paths'], starts, goals,
                    title=f"Prioritized Planning (Collisions: {result_pp['collisions']})",
                    filename='demo_prioritized.png')
    
    # CBS
    result_cbs = cbs(grid, starts, goals)
    print(f"\nCBS:")
    print(f"  Sum of Costs: {result_cbs['sum_of_costs']}, Solution Collisions: {result_cbs['collisions']} (all {result_cbs.get('initial_collisions', 0)} initial conflicts successfully resolved)")
    print(f"  CT Nodes Expanded: {result_cbs['ct_nodes_expanded']}")
    plot_multi_agent(grid, result_cbs['paths'], starts, goals,
                    title=f"CBS - Optimal (Sum of Costs: {result_cbs['sum_of_costs']})",
                    filename='demo_cbs.png')
    
    print("\nDemo images saved!")


def run_q1():
    """Run Q1 benchmarks and generate plots."""
    print("\n" + "=" * 60)
    print("Q1: Single-Agent A* Heuristic Comparison")
    print("=" * 60)
    print("Running benchmarks (this may take a few minutes)...")
    
    results = run_q1_benchmarks(
        grid_sizes=[20, 30, 50],
        obstacle_densities=[0.1, 0.2, 0.3],
        num_trials=50
    )
    
    summarize_q1(results)
    
    print("\nGenerating plots...")
    os.makedirs('results/q1', exist_ok=True)
    plot_q1_results(results, output_dir='results/q1')
    print("Q1 complete! Check the generated PNG files in results/q1/.")


def run_q2():
    """Run Q2 benchmarks and generate plots."""
    print("\n" + "=" * 60)
    print("Q2: Multi-Agent Pathfinding Comparison")
    print("=" * 60)
    print("Running benchmarks (this may take several minutes)...")
    
    results = run_q2_benchmarks(
        grid_sizes=[10, 15],
        agent_counts=[2, 3, 5, 6],
        obstacle_densities=[0.1, 0.2],
        num_trials=10
    )
    
    summarize_q2(results)
    
    print("\nGenerating plots...")
    os.makedirs('results/q2', exist_ok=True)
    plot_q2_results(results, output_dir='results/q2')
    print("Q2 complete! Check the generated PNG files in results/q2/.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("CSMI17 - Artificial Intelligence Assignment")
    print("Robot Path-finding using A* Search")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python main.py --demo    Run visual demo")
        print("  python main.py --q1      Run Q1 benchmarks (heuristic comparison)")
        print("  python main.py --q2      Run Q2 benchmarks (multi-agent comparison)")
        print("  python main.py --all     Run everything")
        return
    
    arg = sys.argv[1].lower()
    
    if arg == '--demo':
        run_demo()
    elif arg == '--q1':
        run_q1()
    elif arg == '--q2':
        run_q2()
    elif arg == '--all':
        run_demo()
        run_q1()
        run_q2()
    else:
        print(f"Unknown argument: {arg}")
        print("Use --demo, --q1, --q2, or --all")


if __name__ == '__main__':
    main()
