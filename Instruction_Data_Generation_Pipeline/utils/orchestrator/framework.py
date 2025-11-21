"""
MCTS Instruction Data Generator: Main framework for MCTS-driven hierarchical reasoning path discovery.

This class orchestrates the three iterative phases:
1) Selection: Select nodes to expand based on UCB strategy
2) Expansion & Evaluation: Generate candidate children and assess quality
3) Backpropagation: Propagate evaluation scores up the reasoning tree
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from ..tree import MCTSTree
from ..nodes import (
    MCTSNode,
)
from ..services.qa.generator import QAGenerator
from ..services.qa.evaluator import QAEvaluator
from ..formatter import ShareGPTConverter
from ..batch import ParallelBatchManager
from ..tree.expansion_controller import TreeExpansionController

# Helpers (private modules)
from ._iterations import run_mcts_iterations
from ._expansion import (
    expand_node_batch,
    expand_multiple_nodes_parallel,
)
from ._reporting import (
    get_all_complete_paths,
    generate_final_report,
)


class MCTSInstructionDataGenerator:
    """
    MCTS-driven instruction data generation framework for hierarchical reasoning path discovery.
    
    Implements the three iterative phases:
    1) Selection: Select nodes to expand based on UCB strategy
    2) Expansion & Evaluation: Generate candidate children and assess quality
    3) Backpropagation: Propagate evaluation scores up the reasoning tree
    """

    def __init__(
        self,
        client,
        config: dict,
        shared_semaphore=None,
    ) -> None:
        self.client = client
        self.config = config or {}
        
        # Read tunables from config
        mcts_cfg = self.config.get('mcts', {})
        self.max_iterations = int(mcts_cfg.get('max_iterations', 1000))
        self.max_depth = int(mcts_cfg.get('max_depth', 3))
        self.prune_interval = int(mcts_cfg.get('prune_interval', 10))
        parallel_cfg = self.config.get('parallel', {})
        self.node_parallel_workers = int(parallel_cfg.get('nodes', 1))
        self.batch_parallel_workers = int(parallel_cfg.get('batchs', 1))

        # Initialize core components
        exploration = float(mcts_cfg.get('exploration_constant', 1.41))
        self.tree = MCTSTree(
            max_depth=self.max_depth,
            exploration_constant=exploration,
            config=self.config,
        )
        
        # Create QA components with simplified interface
        self.qa_generator = QAGenerator(client, config=self.config)
        self.qa_evaluator = QAEvaluator(client, config=self.config)
        
        # Read qa_mode from config, default to "multiple_choice"
        run_cfg = self.config.get('run', {})
        qa_mode = run_cfg.get('qa_mode', 'multiple_choice')
        effective_seed = run_cfg.get('random_seed')
        self.converter = ShareGPTConverter(seed=effective_seed, mode=qa_mode)

        # Initialize components
        self.batch_manager = ParallelBatchManager(
            qa_generator=self.qa_generator,
            qa_evaluator=self.qa_evaluator,
            config=self.config,
            shared_semaphore=shared_semaphore,
        )
        self.tree_controller = TreeExpansionController(self.config)

        # Statistics and monitoring
        self.stats: Dict[str, Any] = {
            "iterations_completed": 0,
            "nodes_generated": 0,
            "successful_expansions": 0,
            "failed_generations": 0,
            "failed_evaluations": 0,
            "nodes_pruned": 0,
            "total_api_calls": 0,
            "start_time": None,
            "end_time": None,
        }

        self.logger = logging.getLogger(__name__)

    # ---------------------------- Public API ---------------------------- #
    def generate_instruction_data(
        self,
        image_path: str,
        output_path: str,
        tree_save_path: str = None,
        top_k_paths: int = None,
    ) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting MCTS-driven reasoning path discovery for image: {image_path}")
            self.stats["start_time"] = time.time()

            # Run MCTS iterations (root will be expanded first if tree is empty)
            self.logger.info(f"Running {self.max_iterations} MCTS iterations...")
            self._run_mcts_iterations(image_path)

            # Extract best paths
            if top_k_paths is None:
                run_cfg = self.config.get('run', {})
                top_k_paths = int(run_cfg.get('top_k_paths', 20))
            self.logger.info(f"Extracting top {top_k_paths} reasoning paths...")
            best_paths = self.tree.get_best_paths(top_k_paths)
            if not best_paths:
                self.logger.warning("No complete paths found, using available paths")
                all_paths = self._get_all_complete_paths()
                best_paths = all_paths[:top_k_paths]

            # Convert to ShareGPT format
            self.logger.info("Converting paths to ShareGPT format...")
            sharegpt_data = self.converter.convert_paths_to_sharegpt(best_paths, image_path, self.tree)

            # Save results
            if sharegpt_data:
                if output_path:
                    self.converter.save_sharegpt_data(sharegpt_data, output_path)
                    self.logger.info(f"Successfully saved {len(sharegpt_data)} conversations to {output_path}")
                else:
                    self.logger.info(f"Generated {len(sharegpt_data)} conversations (not saved, output_path=None)")
            else:
                self.logger.warning("No ShareGPT conversations generated")

            if tree_save_path:
                self.tree.save_tree_state(tree_save_path)
                self.logger.info(f"Tree state saved to {tree_save_path}")

            self.stats["end_time"] = time.time()
            self.stats["total_api_calls"] = getattr(self.client, "call_count", 0)

            return self._generate_final_report(sharegpt_data, best_paths)
        except Exception as exc:
            self.logger.error(f"Framework execution failed: {exc}")
            raise

    def load_existing_tree(self, tree_path: str) -> None:
        self.tree.load_tree_state(tree_path)
        self.logger.info(f"Loaded existing tree with {len(self.tree.nodes)} nodes")

    def get_current_statistics(self) -> Dict[str, Any]:
        tree_stats = self.tree.get_statistics()
        return {
            "current_stats": self.stats.copy(),
            "tree_stats": tree_stats.__dict__,
            "client_calls": getattr(self.client, "call_count", 0),
        }

    # ---------------------------- Delegates ----------------------------- #
    def _run_mcts_iterations(self, image_path: str) -> None:
        run_mcts_iterations(self, image_path)

    def _expand_node_batch(self, parent_node: MCTSNode, target_count: int, image_path: str) -> List[MCTSNode]:
        return expand_node_batch(self, parent_node, target_count, image_path)

    def _expand_multiple_nodes_parallel(
        self,
        expansion_targets: List[Tuple[MCTSNode, int]],
        image_path: str,
    ) -> int:
        return expand_multiple_nodes_parallel(self, expansion_targets, image_path)

    def _get_all_complete_paths(self) -> List[List[MCTSNode]]:
        return get_all_complete_paths(self)

    def _generate_final_report(self, sharegpt_data: List[Dict[str, Any]], paths: List[List[MCTSNode]]) -> Dict[str, Any]:
        return generate_final_report(self, sharegpt_data, paths)
