"""
Expansion helpers used by the orchestrator.
"""

from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..nodes import MCTSNode


def expand_node_batch(
    framework, parent_node: MCTSNode, target_count: int, image_path: str
) -> List[MCTSNode]:
    expanded_nodes = framework.batch_manager.expand_node_batch(
        parent_node=parent_node,
        target_count=target_count,
        image_path=image_path,
    )

    if expanded_nodes:
        _dup_cfg = getattr(framework, "config", {}).get("tree", {}).get("duplicate", {})
        if _dup_cfg.get("enabled", True):
            filtered_nodes: List[MCTSNode] = []
            for node in expanded_nodes:
                if framework.tree.duplicate_checker.is_duplicate(node.question, node.answer):
                    continue
                if (
                    framework.tree.duplicate_checker.has_similar_question(node.question, config=getattr(framework, "config", {}))
                    or framework.tree.duplicate_checker.has_similar_answer(node.answer, config=getattr(framework, "config", {}))
                ):
                    continue
                filtered_nodes.append(node)
            expanded_nodes = filtered_nodes
        if not expanded_nodes:
            return []
        qa_data_list = [
            {
                "question": node.question,
                "answer": node.answer,
                "context": node.generation_context,
            }
            for node in expanded_nodes
        ]
        new_children = framework.tree.expand_node_batch(parent_node, qa_data_list)
        for child_node, temp_node in zip(new_children, expanded_nodes):
            child_node.total_reward = temp_node.total_reward
            child_node.visit_count = max(child_node.visit_count, 1)
            framework.tree.backpropagate(child_node, child_node.average_reward)
        framework.stats["nodes_generated"] += len(new_children)
        return new_children
    return []


def expand_multiple_nodes_parallel(
    framework, expansion_targets: List[Tuple[MCTSNode, int]], image_path: str
) -> int:
    framework.logger.info(f"Starting parallel expansion for {len(expansion_targets)} nodes")
    max_node_workers = min(len(expansion_targets), framework.node_parallel_workers)
    total_expansions = 0
    with ThreadPoolExecutor(max_workers=max_node_workers) as executor:
        future_to_node = {
            executor.submit(framework._expand_node_batch, parent_node, target_count, image_path): (
                parent_node,
                target_count,
            )
            for parent_node, target_count in expansion_targets
        }
        for future in as_completed(future_to_node):
            parent_node, target_count = future_to_node[future]
            try:
                expanded_nodes = future.result()
                if expanded_nodes:
                    total_expansions += len(expanded_nodes)
                    framework.stats["successful_expansions"] += len(expanded_nodes)
                    framework.logger.debug(
                        f"Node {parent_node.node_id} expanded with {len(expanded_nodes)} children"
                    )
            except Exception as exc:
                framework.logger.warning(f"Node {parent_node.node_id} expansion failed: {exc}")
    framework.logger.info(f"Parallel expansion completed: total {total_expansions} children")
    return total_expansions
