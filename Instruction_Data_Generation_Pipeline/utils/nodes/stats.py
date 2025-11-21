"""
Tree statistics dataclass.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TreeStatistics:
    total_nodes: int = 0
    nodes_by_level: Dict[int, int] = None
    total_visits: int = 0
    average_reward: float = 0.0
    max_depth: int = 0

    def __post_init__(self):
        if self.nodes_by_level is None:
            self.nodes_by_level = {}

