"""
MCTS-driven Hierarchical Reasoning Tree
Concise class that delegates heavy logic to helper modules.
"""

import math
import threading
from typing import List, Dict, Optional, Any

from ..nodes import (
    MCTSNode,
    TreeStatistics,
    QADuplicateChecker,
)
from . import _paths
from . import _stats
from . import _persistence as _persist
from . import _prune


class MCTSTree:
    """
    MCTS Tree for hierarchical Q&A generation.
    Implements Selection, Expansion, Evaluation, and Backpropagation.
    """

    def __init__(self, max_depth: int = 3, exploration_constant: float = 1.41,
                 config: Optional[Dict[str, Any]] = None):
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.config: Dict[str, Any] = config or {}

        self.nodes: Dict[str, MCTSNode] = {}
        self.root_node: Optional[MCTSNode] = None
        self.duplicate_checker = QADuplicateChecker()

        self.iteration_count = 0
        self.total_expansions = 0
        self.total_evaluations = 0

        self.level_node_counts = {i: 0 for i in range(1, max_depth + 1)}
        self.expansion_history: List[Any] = []
        self.quality_history: List[Any] = []

        self._lock = threading.RLock()
        self._tree_controller = None
        self._batch_manager = None

        self._create_virtual_root()

    # --------------------------- Construction --------------------------- #
    def _create_virtual_root(self) -> None:
        virtual_root = MCTSNode(
            question="ROOT",
            answer="ROOT",
            level=0,
        )
        self.root_node = virtual_root
        self.nodes[virtual_root.node_id] = virtual_root

    def has_content_nodes(self) -> bool:
        return any(node.level > 0 for node in self.nodes.values())

    def add_root_node(self, question: str, answer: str, context: Dict[str, Any] = None) -> MCTSNode:
        with self._lock:
            if self.duplicate_checker.is_duplicate(question, answer):
                raise ValueError("Duplicate Q&A pair detected")
            node = MCTSNode(
                question=question,
                answer=answer,
                level=1,
                generation_context=context or {},
            )
            self.root_node.add_child(node)
            self.nodes[node.node_id] = node
            self.duplicate_checker.add_qa_pair(question, answer)
            if 1 in self.level_node_counts:
                self.level_node_counts[1] += 1
            return node

    # ------------------------------ UCB --------------------------------- #
    def _calculate_ucb_score(self, node: MCTSNode) -> float:
        if node.visit_count == 0:
            return float('inf')
        parent_visits = 1
        if node.parent_id and node.parent_id in self.nodes:
            parent_visits = self.nodes[node.parent_id].visit_count
        elif node.is_root:
            parent_visits = max(1, self.iteration_count)
        parent_visits = max(1, parent_visits)
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(parent_visits) / node.visit_count
        )
        return node.average_reward + exploration_term

    # --------------------------- Selection ------------------------------ #
    def select_node_for_expansion(self) -> Optional[MCTSNode]:
        leaf_nodes = self.get_leaf_nodes()
        if not leaf_nodes:
            return None
        expandable_nodes = [node for node in leaf_nodes if node.level < self.max_depth]
        if not expandable_nodes:
            return None
        best_node = None
        best_score = float('-inf')
        for node in expandable_nodes:
            ucb_score = self._calculate_ucb_score(node)
            if ucb_score > best_score:
                best_score = ucb_score
                best_node = node
        return best_node

    def select_nodes_for_expansion(self, max_selections: int = 3, config: dict = None):
        # Delegate to selector with controller.
        from .expansion_controller import TreeExpansionController
        from ..batch import NodeSelector
        controller = TreeExpansionController(config or self.config)
        selector = NodeSelector(controller)
        return selector.select_nodes_for_expansion(self, max_selections)

    # --------------------------- Expansion ------------------------------ #
    def expand_node(self, parent_node: MCTSNode, question: str, answer: str,
                    reward: float, context: Dict[str, Any] = None) -> Optional[MCTSNode]:
        with self._lock:
            if self.duplicate_checker.is_duplicate(question, answer):
                return None
            if parent_node.level >= self.max_depth:
                return None
            child_node = MCTSNode(
                question=question,
                answer=answer,
                level=parent_node.level + 1,
                generation_context=context or {},
            )
            parent_node.add_child(child_node)
            self.nodes[child_node.node_id] = child_node
            self.duplicate_checker.add_qa_pair(question, answer)
            child_node.visit_count = 1
            child_node.total_reward = reward
            self.total_expansions += 1
            if child_node.level in self.level_node_counts:
                self.level_node_counts[child_node.level] += 1
            return child_node

    def expand_node_batch(self, parent_node: MCTSNode, qa_data_list: List[Dict[str, Any]]) -> List[MCTSNode]:
        new_children = []
        with self._lock:
            for qa_data in qa_data_list:
                if self.duplicate_checker.is_duplicate(qa_data['question'], qa_data['answer']):
                    continue
                if parent_node.level >= self.max_depth:
                    continue
                child_node = MCTSNode(
                    question=qa_data['question'],
                    answer=qa_data['answer'],
                    level=parent_node.level + 1,
                    generation_context=qa_data.get('context', {}),
                )
                parent_node.add_child(child_node)
                self.nodes[child_node.node_id] = child_node
                self.duplicate_checker.add_qa_pair(qa_data['question'], qa_data['answer'])
                quality_score = qa_data.get('quality_score', 0.5)
                child_node.visit_count = 1
                child_node.total_reward = quality_score
                new_children.append(child_node)
                self.total_expansions += 1
                if child_node.level in self.level_node_counts:
                    self.level_node_counts[child_node.level] += 1
        return new_children

    # ------------------------- Backpropagation -------------------------- #
    def evaluate_node(self, node: MCTSNode, reward: float) -> None:
        with self._lock:
            node.visit_count += 1
            node.total_reward += reward
            self.total_evaluations += 1

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            self.evaluate_node(current, reward)
            parent = self.nodes.get(current.parent_id) if current.parent_id else None
            current = parent

    # --------------------------- Paths/utils --------------------------- #
    def _find_lowest_common_ancestor(self, node1: MCTSNode, node2: MCTSNode) -> Optional[MCTSNode]:
        return _paths.find_lowest_common_ancestor(node1, node2, self.nodes)

    def calculate_same_level_tree_distance(self, node1: MCTSNode, node2: MCTSNode) -> int:
        return _paths.same_level_tree_distance(node1, node2, self.nodes)

    def find_same_level_nodes_by_distance(self, target_node: MCTSNode, max_distance: int = 10) -> Dict[int, List[MCTSNode]]:
        return _paths.find_same_level_nodes_by_distance(self, target_node, max_distance)

    def _collect_paths(self, node: MCTSNode, current_path: List[MCTSNode], all_paths: List[List[MCTSNode]]):
        _paths.collect_paths(node, current_path, all_paths)

    def _get_all_paths(self) -> List[List[MCTSNode]]:
        return _paths.get_all_paths(self)

    def get_best_paths(self, top_k: int = 10) -> List[List[MCTSNode]]:
        return _paths.get_best_paths(self, top_k)

    # ----------------------------- Accessors --------------------------- #
    def get_leaf_nodes(self) -> List[MCTSNode]:
        return _stats.get_leaf_nodes(self)

    def get_nodes_by_level(self, level: int) -> List[MCTSNode]:
        return _stats.get_nodes_by_level(self, level)

    # ----------------------------- Metrics ----------------------------- #
    def get_statistics(self) -> TreeStatistics:
        return _stats.get_statistics(self)

    def get_structure_stats(self, config: dict = None) -> Dict[str, Any]:
        return _stats.get_structure_stats(self, config)

    # --------------------------- Maintenance --------------------------- #
    def prune_low_quality_nodes(self, min_visits: int = 2, min_reward: float = 0.3, config: dict = None) -> int:
        return _prune.prune_low_quality_nodes(self, min_visits, min_reward, config)

    def save_tree_state(self, filepath: str) -> None:
        _persist.save_tree_state(self, filepath)

    def get_tree_state(self) -> dict:
        """Get the current tree state as a dictionary (without saving to file)"""
        return _persist.get_tree_state(self)

    def load_tree_state(self, filepath: str) -> None:
        _persist.load_tree_state(self, filepath)

    def clear(self) -> None:
        with self._lock:
            self.nodes.clear()
            self.root_node = None
            self.duplicate_checker.clear()
            self.iteration_count = 0
            self.total_expansions = 0
            self.total_evaluations = 0
            self.level_node_counts = {i: 0 for i in range(1, self.max_depth + 1)}
            self._create_virtual_root()
