"""
Pruning utilities for the MCTS tree.
"""


def prune_low_quality_nodes(tree, min_visits: int = 2, min_reward: float = 0.3, config: dict = None) -> int:
    # Get pruning config if provided
    if config:
        pruning_config = config.get("tree", {}).get("pruning", {})
        min_visits = pruning_config.get("min_visits", min_visits)
        min_reward = pruning_config.get("min_reward", min_reward)
    with tree._lock:
        prune_roots = []
        for node_id, node in list(tree.nodes.items()):
            if node.is_root:
                continue
            if (node.visit_count < min_visits or (node.visit_count > 0 and node.average_reward < min_reward)):
                prune_roots.append(node_id)
        if not prune_roots:
            return 0
        nodes_to_remove: set = set()

        def collect_descendants(nid: str):
            if nid in nodes_to_remove:
                return
            nodes_to_remove.add(nid)
            node = tree.nodes.get(nid)
            if not node:
                return
            for child in list(node.children):
                collect_descendants(child.node_id)

        for root_id in prune_roots:
            collect_descendants(root_id)

        for node_id in list(nodes_to_remove):
            node = tree.nodes.get(node_id)
            if not node:
                continue
            parent_id = node.parent_id
            if parent_id and parent_id in tree.nodes and parent_id not in nodes_to_remove:
                parent = tree.nodes[parent_id]
                parent.children = [c for c in parent.children if c.node_id not in nodes_to_remove]

        removed_count = 0
        for node_id in list(nodes_to_remove):
            node = tree.nodes.get(node_id)
            if not node:
                continue
            if node.level in tree.level_node_counts and node.level > 0:
                tree.level_node_counts[node.level] = max(0, tree.level_node_counts[node.level] - 1)
            del tree.nodes[node_id]
            removed_count += 1

        tree.duplicate_checker.clear()
        for node in tree.nodes.values():
            if node.level > 0:
                tree.duplicate_checker.add_qa_pair(node.question, node.answer)
        return removed_count

