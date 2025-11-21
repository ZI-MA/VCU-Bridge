"""
MCTS node dataclass and helpers.
"""

import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MCTSNode:
    """
    Represents a single node in the MCTS tree.
    Each node contains a question-answer pair and MCTS metadata.
    """

    # Core QA content
    question: str
    answer: str

    # MCTS metadata
    node_id: Optional[str] = None
    parent_id: Optional[str] = None
    level: int = 1
    visit_count: int = 0
    total_reward: float = 0.0

    # Tree navigation
    children: List["MCTSNode"] = None

    # Additional metadata
    created_timestamp: Optional[float] = None
    generation_context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.node_id is None:
            self.node_id = str(uuid.uuid4())
        if self.created_timestamp is None:
            import time
            self.created_timestamp = time.time()
        if self.generation_context is None:
            self.generation_context = {}

    @property
    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def add_child(self, child_node: "MCTSNode") -> None:
        child_node.parent_id = self.node_id
        self.children.append(child_node)

    def get_sibling_nodes(self, all_nodes: Dict[str, "MCTSNode"]) -> List["MCTSNode"]:
        if self.parent_id is None:
            return []
        siblings = []
        for node in all_nodes.values():
            if (
                node.parent_id == self.parent_id
                and node.node_id != self.node_id
                and node.level == self.level
            ):
                siblings.append(node)
        return siblings

    def get_path_to_root(self, all_nodes: Dict[str, "MCTSNode"]) -> List["MCTSNode"]:
        path = [self]
        current = self
        while current.parent_id is not None:
            parent = all_nodes.get(current.parent_id)
            if parent is None:
                break
            path.append(parent)
            current = parent
        return path[::-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "level": self.level,
            "question": self.question,
            "answer": self.answer,
            "visit_count": self.visit_count,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "created_timestamp": self.created_timestamp,
            "generation_context": self.generation_context,
            "children_ids": [child.node_id for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSNode":
        node = cls(
            question=data["question"],
            answer=data["answer"],
            node_id=data["node_id"],
            parent_id=data.get("parent_id"),
            level=data.get("level", 1),
            visit_count=data.get("visit_count", 0),
            total_reward=data.get("total_reward", 0.0),
            created_timestamp=data.get("created_timestamp"),
            generation_context=data.get("generation_context", {}),
        )
        return node

