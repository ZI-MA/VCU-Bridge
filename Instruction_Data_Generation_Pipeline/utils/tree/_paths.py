"""
Path and distance helpers for the MCTS tree.
"""

from typing import Dict, List, Optional

from ..nodes import MCTSNode


def find_lowest_common_ancestor(node1: MCTSNode, node2: MCTSNode, nodes: Dict[str, MCTSNode]) -> Optional[MCTSNode]:
    path1 = node1.get_path_to_root(nodes)
    path2 = node2.get_path_to_root(nodes)
    lca = None
    for a, b in zip(path1, path2):
        if a.node_id == b.node_id:
            lca = a
        else:
            break
    return lca


def same_level_tree_distance(node1: MCTSNode, node2: MCTSNode, nodes: Dict[str, MCTSNode]) -> int:
    if node1.level != node2.level:
        raise ValueError("Nodes must be at the same level")
    if node1.node_id == node2.node_id:
        return 0
    lca = find_lowest_common_ancestor(node1, node2, nodes)
    if lca is None:
        return float('inf')
    path1 = node1.get_path_to_root(nodes)
    path2 = node2.get_path_to_root(nodes)
    path1_reversed = list(reversed(path1))
    path2_reversed = list(reversed(path2))
    distance1 = 0
    distance2 = 0
    for n in path1_reversed:
        if n.node_id == lca.node_id:
            break
        distance1 += 1
    for n in path2_reversed:
        if n.node_id == lca.node_id:
            break
        distance2 += 1
    return distance1 + distance2


def find_same_level_nodes_by_distance(tree, target_node: MCTSNode, max_distance: int = 10) -> Dict[int, List[MCTSNode]]:
    same_level_nodes = tree.get_nodes_by_level(target_node.level)
    nodes_by_distance: Dict[int, List[MCTSNode]] = {}
    for node in same_level_nodes:
        if node.node_id == target_node.node_id:
            continue
        try:
            distance = same_level_tree_distance(target_node, node, tree.nodes)
            if distance <= max_distance:
                nodes_by_distance.setdefault(distance, []).append(node)
        except ValueError:
            continue
    return nodes_by_distance


def collect_paths(node: MCTSNode, current_path: List[MCTSNode], all_paths: List[List[MCTSNode]]):
    current_path.append(node)
    if node.is_leaf:
        all_paths.append(current_path.copy())
    else:
        for child in node.children:
            collect_paths(child, current_path, all_paths)
    current_path.pop()


def get_all_paths(tree) -> List[List[MCTSNode]]:
    if not tree.root_node:
        return []
    all_paths: List[List[MCTSNode]] = []
    for root_child in tree.root_node.children:
        collect_paths(root_child, [], all_paths)
    return all_paths


def get_best_paths(tree, top_k: int = 10) -> List[List[MCTSNode]]:
    """Get the best paths from the tree based on average reward"""
    # Collect leaf nodes at max_depth only
    leaf_nodes = [node for node in tree.nodes.values() 
                  if node.level > 0 and node.is_leaf and node.level == tree.max_depth]
    if not leaf_nodes:
        return []
    
    # Score complete paths by average reward across content nodes
    path_scores: List[tuple] = []
    for leaf in leaf_nodes:
        full_path = leaf.get_path_to_root(tree.nodes)
        content_path = [n for n in full_path if n.level > 0]
        if len(content_path) == tree.max_depth:
            score = sum(n.average_reward for n in content_path) / len(content_path)
            path_scores.append((score, full_path))
    
    path_scores.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in path_scores[:top_k]]

