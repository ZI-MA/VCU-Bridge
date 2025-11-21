"""
Unified MCTS iteration logic.
"""

from typing import List, Tuple
from tqdm import tqdm

from ..nodes import MCTSNode


def run_mcts_iterations(framework, image_path: str) -> None:
    """Unified MCTS iteration logic for reasoning path discovery."""
    with tqdm(total=framework.max_iterations, desc="Reasoning Path Discovery") as pbar:
        for iteration in range(framework.max_iterations):
            framework.stats["iterations_completed"] = iteration + 1
            expansion_targets: List[Tuple[MCTSNode, int]] = framework.tree.select_nodes_for_expansion(
                max_selections=framework.node_parallel_workers,
                config=framework.config,
            )

            if not expansion_targets:
                framework.logger.info(f"No expandable nodes at iteration {iteration}")
                break

            # Use parallel processing logic for all nodes
            total_expansions = framework._expand_multiple_nodes_parallel(expansion_targets, image_path)

            if total_expansions == 0:
                framework.stats["failed_generations"] += 1

            framework.tree.iteration_count += 1

            if iteration % framework.prune_interval == 0 and iteration > 0:
                pruned_count = framework.tree.prune_low_quality_nodes(config=framework.config)
                framework.stats["nodes_pruned"] += pruned_count
                if pruned_count > 0:
                    framework.logger.info(f"Pruned {pruned_count} low-quality nodes")

            struct_stats = framework.tree.get_structure_stats()
            pbar.set_postfix(
                {
                    "Nodes": len(framework.tree.nodes) - 1,
                    "Expansions": total_expansions,
                    "AvgBranch": f"{struct_stats['branching_factor']['average']:.1f}",
                    "Balance": f"{struct_stats['balance_info'].get('balance_score', 0.0):.2f}",
                }
            )
            pbar.update(1)

    framework.logger.info(
        f"MCTS iterations completed: {framework.stats['iterations_completed']}"
    )
