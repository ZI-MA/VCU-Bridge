"""
Node selection for expansion.
"""

from typing import List, Tuple
import logging

from ..nodes import MCTSNode
from ..tree.expansion_controller import TreeExpansionController


class NodeSelector:
    def __init__(self, tree_controller: TreeExpansionController):
        self.tree_controller = tree_controller
        self.logger = logging.getLogger(__name__)

    def select_nodes_for_expansion(self, tree, max_selections: int = 3) -> List[Tuple[MCTSNode, int]]:
        # Protect selection process with lock to prevent race conditions
        with self.tree_controller._expansion_lock:
            expandable_nodes = [node for node in tree.get_leaf_nodes() if node.level < tree.max_depth]
            if not expandable_nodes:
                self.logger.info("No expandable nodes found")
                return []

            # Step 1: Calculate UCB scores and sort (without allocating)
            scored_nodes: List[Tuple[float, MCTSNode]] = []
            for node in expandable_nodes:
                if not self._should_expand_node(node, tree):
                    continue
                ucb_score = tree._calculate_ucb_score(node)
                scored_nodes.append((ucb_score, node))

            if not scored_nodes:
                return []

            # Sort by UCB score, descending priority
            scored_nodes.sort(key=lambda x: x[0], reverse=True)

            # Step 2: Allocate expansion count by priority
            candidates: List[Tuple[MCTSNode, int]] = []
            allocated_per_level = {}  # Track allocated count per level
            allocated_total = 0       # Track total allocated count

            selected_count = min(max_selections, len(scored_nodes))
            for i in range(selected_count):
                _, node = scored_nodes[i]
                try:
                    expansion_count = self.tree_controller.get_expansion_count(
                        node, tree, allocated_per_level, allocated_total
                    )
                    if expansion_count > 0:
                        candidates.append((node, expansion_count))
                        # Update allocated counts
                        target_level = node.level + 1
                        allocated_per_level[target_level] = allocated_per_level.get(target_level, 0) + expansion_count
                        allocated_total += expansion_count
                except Exception:
                    # Return minimum value on failure
                    if allocated_total < self.tree_controller.max_total_nodes:
                        candidates.append((node, 1))
                        allocated_total += 1

            return candidates

    def _should_expand_node(self, node: MCTSNode, tree) -> bool:
        try:
            return self.tree_controller.can_expand(node, tree)
        except Exception:
            return node.level < tree.max_depth
