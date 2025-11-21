"""
Reporting helpers for final results and path collection.
"""

from typing import Any, Dict, List


def get_all_complete_paths(framework) -> List[list]:
    tree = framework.tree
    return tree._get_all_paths() if hasattr(tree, "_get_all_paths") else []


def generate_final_report(
    framework,
    sharegpt_data: List[Dict[str, Any]],
    paths: List[list],
) -> Dict[str, Any]:
    tree_stats = framework.tree.get_statistics()
    struct_stats = framework.tree.get_structure_stats()
    conversion_stats = framework.converter.get_conversion_statistics(sharegpt_data)
    execution_time = 0.0
    if framework.stats["start_time"] and framework.stats["end_time"]:
        execution_time = framework.stats["end_time"] - framework.stats["start_time"]

    report: Dict[str, Any] = {
        "execution_summary": {
            "total_execution_time_seconds": execution_time,
            "iterations_completed": framework.stats["iterations_completed"],
            "max_iterations": framework.max_iterations,
            "completion_rate": framework.stats["iterations_completed"] / max(1, framework.max_iterations),
            "total_api_calls": framework.stats["total_api_calls"],
            "node_parallel_workers": framework.node_parallel_workers,
            "batch_parallel_workers": framework.batch_parallel_workers,
        },
        "tree_statistics": {
            "total_nodes_generated": framework.stats["nodes_generated"],
            "total_nodes_final": tree_stats.total_nodes,
            "successful_expansions": framework.stats["successful_expansions"],
            "failed_generations": framework.stats["failed_generations"],
            "failed_evaluations": framework.stats["failed_evaluations"],
            "nodes_pruned": framework.stats["nodes_pruned"],
            "nodes_by_level": tree_stats.nodes_by_level,
            "average_reward": tree_stats.average_reward,
        },
        "path_extraction": {
            "total_paths_extracted": len(paths),
            "average_path_length": (sum(len(p) - 1 for p in paths) / len(paths)) if paths else 0,
            "max_path_length": max((len(p) - 1 for p in paths), default=0),
        },
        "sharegpt_conversion": conversion_stats,
        "quality_metrics": {
            "expansion_success_rate": (
                framework.stats["successful_expansions"]
                / max(1, framework.stats["successful_expansions"] + framework.stats["failed_generations"])
            ),
            "evaluation_success_rate": (
                (framework.stats["successful_expansions"] + framework.stats["failed_generations"]) /
                max(
                    1,
                    framework.stats["failed_evaluations"]
                    + framework.stats["successful_expansions"]
                    + framework.stats["failed_generations"],
                )
            ),
        },
    }

    if struct_stats:
        report["structure_statistics"] = struct_stats
        report["tree_structure"] = {
            "average_branching_factor": struct_stats["branching_factor"]["average"],
            "balance_score": struct_stats["balance_info"]["balance_score"],
            "level_utilization": struct_stats["level_utilization"],
            "tree_shape_analysis": struct_stats["balance_info"].get("recommendations", []),
        }

    if framework.batch_manager:
        batch_stats = framework.batch_manager.get_generation_statistics()
        report["batch_expansion_stats"] = batch_stats

    report["sharegpt_data"] = sharegpt_data or []
    
    if hasattr(framework.tree, 'get_tree_state'):
        report["tree_state"] = framework.tree.get_tree_state()

    return report
