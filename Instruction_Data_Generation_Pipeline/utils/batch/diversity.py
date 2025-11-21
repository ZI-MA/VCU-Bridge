"""
Sibling diversity checking utilities.
"""

from typing import List
from difflib import SequenceMatcher
import logging

from ..nodes import MCTSNode


class SiblingDiversityChecker:
    def __init__(self, max_similarity: float = 0.4, config: dict = None):
        self.config = config or {}
        # Read from tree.diversity.max_similarity
        try:
            self.max_similarity = float(
                self.config.get("tree", {})
                .get("diversity", {})
                .get("max_similarity", max_similarity)
            )
        except Exception:
            self.max_similarity = max_similarity
        self.logger = logging.getLogger(__name__)

    def check_sibling_diversity(self, new_node: MCTSNode, siblings: List[MCTSNode]) -> bool:
        for sibling in siblings:
            similarity = self._calculate_semantic_similarity(new_node, sibling)
            if similarity > self.max_similarity:
                self.logger.debug(
                    f"Sibling similarity too high: {similarity:.3f} > {self.max_similarity:.3f}"
                )
                return False
        return True

    def _calculate_semantic_similarity(self, node1: MCTSNode, node2: MCTSNode) -> float:
        try:
            text1 = f"{node1.question} {node1.answer}".lower()
            text2 = f"{node2.question} {node2.answer}".lower()
            similarity = SequenceMatcher(None, text1, text2).ratio()
            return similarity
        except Exception as e:
            self.logger.warning(f"Similarity computation failed: {e}")
            return 0.0

    def get_diversity_score(self, candidate: MCTSNode, existing_nodes: List[MCTSNode]) -> float:
        if not existing_nodes:
            return 1.0
        similarities = []
        for existing in existing_nodes:
            similarity = self._calculate_semantic_similarity(candidate, existing)
            similarities.append(similarity)
        max_similarity = max(similarities)
        diversity_score = 1.0 - max_similarity
        return diversity_score
