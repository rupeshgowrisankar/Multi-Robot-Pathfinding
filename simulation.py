import argparse
import heapq
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from grid import Grid
from heuristics import HEURISTICS, manhattan
from astar import astar
from prioritized_planning import independent_astar, prioritized_planning, detect_collisions
from cbs import cbs
from hybrid_pp_cbs import hybrid_pp_cbs


# ============================================================
# Common drawing utilities
# ============================================================

def draw_grid(ax, grid):
    ax.clear()
    ax.set_xlim(0, grid.width)
    ax.set_ylim(grid.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(grid.width + 1))
    ax.set_yticks(range(grid.height + 1))
    ax.grid(True, linewidth=0.4)

    # Obstacles
    for x, y in grid.obstacles:
        ax.add_patch(
            plt.Rectangle((x, y), 1, 1, facecolor="black", edgecolor="black")
        )

    # Start / goal
    ax.add_patch(
        plt.Rectangle(
            (grid.start[0], grid.start[1]), 1, 1,
            facecolor="green", edgecolor="black"
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (grid.goal[0], grid.goal[1]), 1, 1,
            facecolor="red", edgecolor="black"
        )
    )

    ax.text(
        grid.start[0] + 0.5, grid.start[1] + 0.5, "S",
        ha="center", va="center", color="white", weight="bold"
    )
    ax.text(
        grid.goal[0] + 0.5, grid.goal[1] + 0.5, "G",
        ha="center", va="center", color="white", weight="bold"
    )


def draw_path(ax, path, color="dodgerblue", linewidth=3, alpha=0.9):
    if not path:
        return

    xs = [x + 0.5 for x, y in path]
    ys = [y + 0.5 for x, y in path]

    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(xs, ys, color=color, s=18, zorder=5)


# ============================================================
# Q1: TRACEABLE A*
# ============================================================


def traced_astar(grid, start, goal, heuristic):
    """
    A* implementation specifically for visualization.

    Returns:
        path
        expanded_order: nodes as they are removed from OPEN
        frontier_history: OPEN contents after every expansion
    """

    counter = 0
    h0 = heuristic(start, goal)

    # (f, h, counter, node)
    open_list = [(h0, h0, counter, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    expanded_order = []
    frontier_history = []

    while open_list:
        f, h, _, current = heapq.heappop(open_list)

        if current in closed:
            continue

        closed.add(current)
        expanded_order.append(current)

        # Store the currently visible frontier.
        frontier_history.append(
            [item[3] for item in open_list if item[3] not in closed]
        )

        if current == goal:
            path = []
            node = goal

            while node in came_from:
                path.append(node)
                node = came_from[node]

            path.append(start)
            path.reverse()

            return {
                "path": path,
                "expanded": expanded_order,
                "frontier": frontier_history,
                "cost": g_score[goal],
            }

        for neighbor in grid.get_neighbors(current):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                h_val = heuristic(neighbor, goal)
                f_val = tentative_g + h_val

                counter += 1
                heapq.heappush(
                    open_list,
                    (f_val, h_val, counter, neighbor)
                )

    return {
        "path": None,
        "expanded": expanded_order,
        "frontier": frontier_history,
        "cost": -1,
    }


def simulate_q1_heuristic(grid, heuristic_name, interval=100):
    """
    Animate A* node expansion for one heuristic.
    """

    heuristic = HEURISTICS[heuristic_name]
    result = traced_astar(grid, grid.start, grid.goal, heuristic)

    expanded = result["expanded"]
    path = result["path"]

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        draw_grid(ax, grid)

        # Nodes already expanded
        visited = expanded[:frame + 1]

        for x, y in visited:
            if (x, y) != grid.start and (x, y) != grid.goal:
                ax.add_patch(
                    plt.Rectangle(
                        (x, y), 1, 1,
                        facecolor="gold",
                        edgecolor="orange",
                        alpha=0.75
                    )
                )

        # Final path appears only at the end
        if frame >= len(expanded) - 1 and path:
            draw_path(ax, path)

        ax.set_title(
            f"A* — {heuristic_name}\n"
            f"Expanded nodes: {min(frame + 1, len(expanded))} / {len(expanded)}"
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=max(1, len(expanded)),
        interval=interval,
        repeat=False
    )


    plt.tight_layout()
    plt.show()

    print("\nQ1 Simulation")
    print("----------------------------")
    print("Heuristic:", heuristic_name)
    print("Start:", grid.start)
    print("Goal:", grid.goal)
    print("Path cost:", result["cost"])
    print("Nodes expanded:", len(expanded))

    return anim


def simulate_q1_all(interval=70):
    """
    Run four separate Q1 simulations sequentially.

    The four heuristics are:
        Manhattan
        Euclidean
        Chebyshev
        Zero (Dijkstra)
    """

    grid = Grid(15, 15, obstacle_density=0.20, seed=42)

    print("\nQ1 GRID")
    print("-------")
    print("Grid:", grid)
    print("Start:", grid.start)
    print("Goal:", grid.goal)

    for name in HEURISTICS:
        print("\nRunning:", name)
        simulate_q1_heuristic(grid, name, interval)


def compare_q1_paths():
    """
    Static side-by-side comparison of final paths for all heuristics.
    """

    grid = Grid(15, 15, obstacle_density=0.20, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    for ax, (name, heuristic) in zip(axes, HEURISTICS.items()):
        result = traced_astar(grid, grid.start, grid.goal, heuristic)

        draw_grid(ax, grid)

        if result["path"]:
            draw_path(ax, result["path"])

        ax.set_title(
            f"{name}\n"
            f"Cost = {result['cost']} | "
            f"Expanded = {len(result['expanded'])}"
        )

    fig.suptitle(
        "Q1 — Final A* Paths for Different Heuristics",
        fontsize=16
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# Q2: MULTI-ROBOT SIMULATION
# ============================================================

AGENT_COLORS = [
    "blue", "orange", "green", "purple",
    "brown", "pink", "cyan", "magenta"
]
AGENT_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def draw_multi_grid(ax, grid, starts, goals):
    ax.clear()
    ax.set_xlim(0, grid.width)
    ax.set_ylim(grid.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(grid.width + 1))
    ax.set_yticks(range(grid.height + 1))
    ax.grid(True, linewidth=0.4)

    for x, y in grid.obstacles:
        ax.add_patch(
            plt.Rectangle(
                (x, y), 1, 1,
                facecolor="black",
                edgecolor="black"
            )
        )

    for i, (start, goal) in enumerate(zip(starts, goals)):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        ax.add_patch(
            plt.Rectangle(
                (start[0], start[1]), 1, 1,
                facecolor=color,
                edgecolor="black"
            )
        )

        ax.add_patch(
            plt.Rectangle(
                (goal[0], goal[1]), 1, 1,
                facecolor=color,
                edgecolor="black",
                alpha=0.35
            )
        )

        ax.text(
            start[0] + 0.5, start[1] + 0.5,
            f"S{i}", ha="center", va="center",
            color="white", weight="bold", fontsize=8
        )

        ax.text(
            goal[0] + 0.5, goal[1] + 0.5,
            f"G{i}", ha="center", va="center",
            color="black", weight="bold", fontsize=8
        )


def plot_multi_paths(grid, starts, goals, paths, title):
    fig, ax = plt.subplots(figsize=(9, 9))

    draw_multi_grid(ax, grid, starts, goals)

    for i, path in enumerate(paths):
        if not path:
            continue

        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ls = AGENT_LINESTYLES[i % len(AGENT_LINESTYLES)]
        xs = [x + 0.5 for x, y in path]
        ys = [y + 0.5 for x, y in path]

        ax.plot(
            xs, ys,
            linestyle=ls,
            color=color,
            linewidth=3,
            label=f"Agent {i}"
        )

        ax.scatter(
            xs, ys,
            color=color,
            s=15
        )

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_multi_paths(grid, starts, goals, paths, title, interval=250):
    """
    Animate robots moving simultaneously along their computed paths.
    """

    max_t = max(len(p) for p in paths)

    fig, ax = plt.subplots(figsize=(9, 9))

    def update(t):
        draw_multi_grid(ax, grid, starts, goals)

        for i, path in enumerate(paths):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            ls = AGENT_LINESTYLES[i % len(AGENT_LINESTYLES)]

            if not path:
                continue

            # Draw complete route faintly
            xs = [x + 0.5 for x, y in path]
            ys = [y + 0.5 for x, y in path]

            ax.plot(
                xs, ys,
                linestyle=ls,
                color=color,
                alpha=0.35
            )

            # Robot waits at goal after reaching it
            index = min(t, len(path) - 1)
            x, y = path[index]

            ax.add_patch(
                plt.Circle(
                    (x + 0.5, y + 0.5),
                    0.28,
                    color=color,
                    zorder=10
                )
            )

            ax.text(
                x + 0.5,
                y + 0.5,
                str(i),
                color="white",
                ha="center",
                va="center",
                weight="bold",
                zorder=11
            )

        ax.set_title(f"{title}\nTime step = {t}")

    anim = FuncAnimation(
        fig,
        update,
        frames=max_t,
        interval=interval,
        repeat=False
    )


    plt.tight_layout()
    plt.show()

    return anim


def _print_result(name, result):
    print(f"\n{name}")
    print("-" * len(name))
    print("Sum of costs:", result["sum_of_costs"])
    print("Makespan:", result["makespan"])
    col_list = result.get("collision_list", [])
    if col_list:
        e_cnt = sum(1 for c in col_list if c['type'] == 'edge')
        v_cnt = sum(1 for c in col_list if c['type'] == 'vertex')
        details = []
        if e_cnt:
            details.append(f"{e_cnt} edge")
        if v_cnt:
            details.append(f"{v_cnt} vertex")
        print(f"Collisions: {result.get('collisions', 0)} ({', '.join(details)})")
    else:
        print("Collisions:", result.get("collisions", 0))
    print("Runtime:", f"{result['time'] * 1000:.2f} ms")
    print("Success:", result["success"])

    if name == "CBS":
        print("Initial unconstrained conflicts:", result.get("initial_collisions", 0))
        print("CT nodes expanded:", result.get("ct_nodes_expanded"))

    if name == "H-PCBS":
        print("Initial PP success:", result.get("initial_pp_success"))
        print("Initial PP collisions:", result.get("initial_collisions"))
        print("CBS repair used:", result.get("used_repair"))
        print("CT nodes expanded:", result.get("ct_nodes_expanded"))
        print("Replans:", result.get("replans"))


def compare_q2(seed=127):
    """Compare all four multi-agent algorithms on the same instance."""
    grid, starts, goals = Grid.generate_multi_agent(
        15, 15, 4,
        obstacle_density=0.20,
        seed=seed
    )

    results = [
        ("Independent A*", independent_astar(
            grid, starts, goals, manhattan)),
        ("Prioritized Planning", prioritized_planning(
            grid, starts, goals, manhattan)),
        ("CBS", cbs(
            grid, starts, goals, manhattan, time_limit=30)),
        ("H-PCBS", hybrid_pp_cbs(
            grid, starts, goals, manhattan, time_limit=30))
    ]

    for name, result in results:
        _print_result(name, result)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for ax, (name, result) in zip(axes, results):
        draw_multi_grid(ax, grid, starts, goals)

        for i, path in enumerate(result["paths"]):
            if not path:
                continue

            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            ls = AGENT_LINESTYLES[i % len(AGENT_LINESTYLES)]
            xs = [x + 0.5 for x, y in path]
            ys = [y + 0.5 for x, y in path]

            ax.plot(
                xs, ys,
                linestyle=ls,
                color=color,
                linewidth=2.5,
                label=f"A{i}"
            )

        ax.set_title(
            f"{name}\n"
            f"SOC = {result['sum_of_costs']} | "
            f"Collisions = {result.get('collisions', 0)}"
        )
        ax.legend(fontsize=7)

    fig.suptitle(
        "Q2 — Four-Algorithm Multi-Robot Comparison",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        "videos/q2_four_algorithm_comparison.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.show()


def simulate_hybrid_q2(seed=127):
    """Demonstrate the proposed H-PCBS algorithm."""
    grid, starts, goals = Grid.generate_multi_agent(
        15, 15, 4,
        obstacle_density=0.20,
        seed=seed
    )

    result = hybrid_pp_cbs(
        grid, starts, goals, manhattan, time_limit=30
    )

    print("\nH-PCBS SIMULATION")
    print("=================")
    print("Grid:", grid)

    for i in range(len(starts)):
        print(f"Agent {i}: {starts[i]} -> {goals[i]}")

    _print_result("H-PCBS", result)

    plot_multi_paths(
        grid,
        starts,
        goals,
        result["paths"],
        f"H-PCBS — SOC={result['sum_of_costs']} | "
        f"Collisions={result['collisions']}"
    )

    animate_multi_paths(
        grid,
        starts,
        goals,
        result["paths"],
        "H-PCBS — Final Solution"
    )


def simulate_q2(seed=127):
    """Demonstrate all four Q2 algorithms."""
    grid, starts, goals = Grid.generate_multi_agent(
        15, 15, 4,
        obstacle_density=0.20,
        seed=seed
    )

    print("\nQ2 MULTI-ROBOT SIMULATION")
    print("==========================")

    methods = [
        ("Independent A*", independent_astar(
            grid, starts, goals, manhattan)),
        ("Prioritized Planning", prioritized_planning(
            grid, starts, goals, manhattan)),
        ("CBS", cbs(
            grid, starts, goals, manhattan, time_limit=30)),
        ("H-PCBS", hybrid_pp_cbs(
            grid, starts, goals, manhattan, time_limit=30))
    ]

    for name, result in methods:
        _print_result(name, result)

        plot_multi_paths(
            grid,
            starts,
            goals,
            result["paths"],
            f"{name} — Collisions={result.get('collisions', 0)}"
        )

        animate_multi_paths(
            grid,
            starts,
            goals,
            result["paths"],
            name
        )


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--q1", action="store_true")
    parser.add_argument("--q1compare", action="store_true")
    parser.add_argument("--q2", action="store_true")
    parser.add_argument("--q2hybrid", action="store_true")
    parser.add_argument("--q2compare", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=127, help="Random seed for multi-agent generation")

    args = parser.parse_args()

    if args.all:
        simulate_q1_all()
        compare_q1_paths()
        simulate_q2(args.seed)
        compare_q2(args.seed)
        return

    if args.q1:
        simulate_q1_all()

    if args.q1compare:
        compare_q1_paths()

    if args.q2:
        simulate_q2(args.seed)

    if args.q2hybrid:
        simulate_hybrid_q2(args.seed)

    if args.q2compare:
        compare_q2(args.seed)

    if not any([
        args.q1, args.q1compare, args.q2,
        args.q2hybrid, args.q2compare, args.all
    ]):
        print("Use:")
        print("  python simulation.py --q1")
        print("  python simulation.py --q1compare")
        print("  python simulation.py --q2")
        print("  python simulation.py --q2hybrid")
        print("  python simulation.py --q2compare")
        print("  python simulation.py --all")


if __name__ == "__main__":
    main()