"""
Statistics helpers for MCTS tree.
"""

from typing import Dict, Any, List, TYPE_CHECKING
import statistics

from ..nodes import TreeStatistics

if TYPE_CHECKING:
    from ..nodes import MCTSNode


def get_statistics(tree) -> TreeStatistics:
    stats = TreeStatistics()
    real_nodes = [n for n in tree.nodes.values() if n.level > 0]
    stats.total_nodes = len(real_nodes)
    stats.nodes_by_level = {i: 0 for i in range(1, tree.max_depth + 1)}
    for n in real_nodes:
        stats.nodes_by_level[n.level] = stats.nodes_by_level.get(n.level, 0) + 1
    stats.total_visits = sum(n.visit_count for n in real_nodes)
    if stats.total_visits > 0:
        total_reward = sum(n.total_reward for n in real_nodes)
        stats.average_reward = total_reward / stats.total_visits
    stats.max_depth = max((n.level for n in real_nodes), default=0)
    return stats


def get_structure_stats(tree, config: dict = None) -> Dict[str, Any]:
    internal_nodes = [n for n in tree.nodes.values() if n.level > 0 and len(n.children) > 0]
    avg_branch = (sum(len(n.children) for n in internal_nodes) / len(internal_nodes)) if internal_nodes else 0.0

    # Calculate balance info directly
    level_counts = {}
    total_nodes = 0
    for level in range(1, tree.max_depth + 1):
        count = len(tree.get_nodes_by_level(level))
        level_counts[level] = count
        total_nodes += count
    
    # Simple balance score using coefficient of variation
    if total_nodes > 0 and len(level_counts) > 1:
        counts = list(level_counts.values())
        mean_count = statistics.mean(counts)
        if mean_count > 0:
            std_dev = statistics.stdev(counts)
            balance_score = max(0.0, 1 - (std_dev / mean_count))
        else:
            balance_score = 1.0
    else:
        balance_score = 1.0 if total_nodes > 0 else 0.0

    balance_info = {
        'level_distribution': level_counts,
        'balance_score': balance_score,
        'recommendations': [],
    }

    # Current distribution ratios by level
    current_distribution: Dict[int, float] = {}
    for level in range(1, tree.max_depth + 1):
        current_distribution[level] = len(tree.get_nodes_by_level(level)) / max(1, total_nodes)

    balance_info['current_distribution'] = current_distribution

    level_utilization: Dict[int, float] = {}
    for level in range(1, tree.max_depth + 1):
        current_count = len(tree.get_nodes_by_level(level))
        tcfg = getattr(tree, 'config', {}) or {}
        lvl_cfg = ((tcfg.get('tree', {}) or {}).get('levels', {}) or {}).get(level, {})
        max_count = int(lvl_cfg.get('max_nodes', 100))
        level_utilization[level] = (current_count / max_count) if max_count > 0 else 0.0

    return {
        "branching_factor": {
            "average": avg_branch,
            "internal_nodes_count": len(internal_nodes),
        },
        "balance_info": balance_info,
        "level_utilization": level_utilization,
    }


def get_leaf_nodes(tree) -> List['MCTSNode']:
    """Get all leaf nodes in the tree.

    When the tree has no content nodes yet (only a virtual root), treat the
    root as expandable to allow level-1 seeding via the normal generator path.
    """
    real_leafs = [node for node in tree.nodes.values() if node.level > 0 and node.is_leaf]
    if real_leafs:
        return real_leafs
    # Fallback for empty trees: include root to kickstart expansion
    root = getattr(tree, 'root_node', None)
    if root is not None and root.is_leaf:
        return [root]
    return []


def get_nodes_by_level(tree, level: int) -> List['MCTSNode']:
    """Get all nodes at a specific level"""
    return [node for node in tree.nodes.values() if node.level == level]
