"""
Persistence helpers for saving/loading tree state.
"""

import json
import time


def get_tree_state(tree) -> dict:
    """Get the current tree state as a dictionary (without saving to file)"""
    with tree._lock:
        return {
            "metadata": {
                "max_depth": tree.max_depth,
                "exploration_constant": tree.exploration_constant,
                "iteration_count": tree.iteration_count,
                "total_expansions": tree.total_expansions,
                "total_evaluations": tree.total_evaluations,
                "timestamp": time.time(),
            },
            "nodes": {node_id: node.to_dict() for node_id, node in tree.nodes.items()},
            "virtual_root_id": tree.root_node.node_id if tree.root_node else None,
        }


def save_tree_state(tree, filepath: str) -> None:
    """Save tree state to file"""
    tree_data = get_tree_state(tree)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)


def load_tree_state(tree, filepath: str) -> None:
    from ..nodes import MCTSNode

    with tree._lock:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        metadata = tree_data["metadata"]
        tree.max_depth = metadata["max_depth"]
        tree.exploration_constant = metadata["exploration_constant"]
        tree.iteration_count = metadata["iteration_count"]
        tree.total_expansions = metadata["total_expansions"]
        tree.total_evaluations = metadata["total_evaluations"]
        tree.nodes = {}
        for node_id, node_data in tree_data["nodes"].items():
            tree.nodes[node_id] = MCTSNode.from_dict(node_data)
        for node in tree.nodes.values():
            if node.parent_id and node.parent_id in tree.nodes:
                parent = tree.nodes[node.parent_id]
                if node not in parent.children:
                    parent.children.append(node)
        virtual_root_id = tree_data.get("virtual_root_id")
        if virtual_root_id and virtual_root_id in tree.nodes:
            tree.root_node = tree.nodes[virtual_root_id]
        else:
            tree._create_virtual_root()
        tree.duplicate_checker.clear()
        for node in tree.nodes.values():
            if node.level > 0:
                tree.duplicate_checker.add_qa_pair(node.question, node.answer)
        tree.level_node_counts = {i: 0 for i in range(1, tree.max_depth + 1)}
        for node in tree.nodes.values():
            if node.level > 0 and node.level in tree.level_node_counts:
                tree.level_node_counts[node.level] += 1

