"""
Batch processing manager with unified parallel logic.
"""

from typing import List, Dict, Optional, Any
import logging
import threading
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from ..nodes import MCTSNode
from .diversity import SiblingDiversityChecker


class ParallelBatchManager:
    """
    Unified batch processing manager. Controls parallelism via max_workers:
    - max_workers = 1: Serial execution (handled internally by ThreadPool)
    - max_workers > 1: True parallel execution
    """
    
    def __init__(self, qa_generator, qa_evaluator, config: dict = None, shared_semaphore=None):
        self.qa_generator = qa_generator
        self.qa_evaluator = qa_evaluator
        self.config = config or {}
        
        # Read all parameters from config
        parallel_cfg = self.config.get('parallel', {})
        client_cfg = self.config.get('client', {})
        tree_cfg = self.config.get('tree', {})
        
        self.max_workers = int(parallel_cfg.get('batchs', 1))
        # Max attempts factor for replacement when rejected by quality/diversity
        self.max_attempts_factor = int(parallel_cfg.get('max_attempts_factor', 3))
        self.enable_caching = bool(parallel_cfg.get('enable_caching', True))
        
        self.diversity_checker = SiblingDiversityChecker(config=self.config)

        # Read acceptance thresholds from config
        self.batch_quality_threshold = float(
            tree_cfg.get('quality', {}).get('acceptance_threshold', 0.6)
        )
        
        # Cache settings
        self.qa_cache = {} if self.enable_caching else None
        self.evaluation_cache = {} if self.enable_caching else None
        self._lock = threading.Lock()
        
        # Concurrency control: prefer shared semaphore if provided
        if shared_semaphore is not None:
            self._client_semaphore = shared_semaphore
        else:
            # Create own semaphore, read from config
            client_max_concurrency = int(client_cfg.get('max_concurrency', max(1, min(self.max_workers, 3))))
            self._client_semaphore = threading.BoundedSemaphore(client_max_concurrency)
        
        # Statistics
        self.stats = {
            'successful_generations': 0,
            'diversity_rejections': 0,
            'quality_rejections': 0,
            'total_attempts': 0
        }
        
        self.parallel_stats = {
            'parallel_calls': 0,
            'cache_hits': 0,
            'total_time_saved': 0.0,
        }
        
        self.logger = logging.getLogger(__name__)

    def expand_node_batch(self, parent_node: MCTSNode, target_count: int,
                         image_path: str) -> List[MCTSNode]:
        start_time = time.time()
        # Optimization: initial workers not exceeding target count
        effective_workers = min(self.max_workers, target_count)
        self.logger.info(
            f"Starting parallel batch expansion for node {parent_node.node_id}, target: {target_count}, workers: {effective_workers} (max: {self.max_workers})"
        )
        # Dynamic replacement: maintain worker pool until target reached or budget exhausted
        successful_nodes: List[MCTSNode] = []
        attempt_budget = max(target_count, target_count * max(1, self.max_attempts_factor))
        attempts = 0
        futures = set()
        # Maintain failure chains: {attempt_id: [failure_reason1, ...]}
        failure_chains = {}
        # Task result tracking: {future: (attempt_id, inherited_chain_id)}
        future_to_task_info = {}

        def add_to_failure_chain(failure_reason: str, attempt_id: int, inherited_chain_id: int = None):
            """Add failure reason to failure chain"""
            if attempt_id is None:
                return
            if inherited_chain_id is not None and inherited_chain_id in failure_chains:
                failure_chains[inherited_chain_id].append(failure_reason)
            else:
                failure_chains[attempt_id] = [failure_reason]

        def submit_one(executor: ThreadPoolExecutor) -> bool:
            nonlocal attempts
            if attempts >= attempt_budget:
                return False

            # Get failure feedback from chain - select existing chain to inherit
            failure_feedback = None
            inherited_chain_id = None
            if failure_chains:
                # Select existing failure chain (simple strategy: shortest chain)
                min_length = min(len(failure_chains[chain_id]) for chain_id in failure_chains)
                for chain_id, chain in failure_chains.items():
                    if len(chain) == min_length:
                        inherited_chain_id = chain_id
                        failure_feedback = "; ".join(chain)
                        break
                if failure_feedback:
                    self.logger.debug(f"Attempt {attempts} inheriting chain {inherited_chain_id}: {failure_feedback[:100]}")

            task = {
                'parent_node': parent_node,
                'image_path': image_path,
                'attempt_id': attempts,
                'failure_feedback': failure_feedback,
                'inherited_chain_id': inherited_chain_id,  # Record inherited chain ID
            }
            fut = executor.submit(self._generate_and_evaluate_parallel, task)
            futures.add(fut)
            future_to_task_info[fut] = (attempts, inherited_chain_id)
            attempts += 1
            # Count attempts
            try:
                self.stats['total_attempts'] += 1
            except Exception:
                pass
            return True

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Fill worker pool first (not exceeding target)
            while len(futures) < effective_workers and attempts < attempt_budget:
                if not submit_one(executor):
                    break

            while futures and len(successful_nodes) < target_count:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.discard(fut)
                    task_info = future_to_task_info.pop(fut, None)
                    if task_info:
                        attempt_id, inherited_chain_id = task_info
                    else:
                        attempt_id, inherited_chain_id = None, None
                    
                    try:
                        result = fut.result()
                        if result:
                            # Check if valid node (has question and answer)
                            if result.question and result.answer and len(successful_nodes) < target_count:
                                if self._check_diversity_parallel(result, successful_nodes + parent_node.children):
                                    successful_nodes.append(result)
                                else:
                                    self.stats['diversity_rejections'] += 1
                                    add_to_failure_chain("Content too similar to existing nodes, lacks diversity", attempt_id, inherited_chain_id)
                            else:
                                # Failed node, add to failure chain
                                failure_reason = getattr(result, 'failure_reason', "Unknown error")
                                add_to_failure_chain(failure_reason, attempt_id, inherited_chain_id)
                        else:
                            # Result is None, add to failure chain
                            add_to_failure_chain("Generation result is empty", attempt_id, inherited_chain_id)
                    except Exception as e:
                        self.logger.warning(f"Parallel task failed: {e}")
                        add_to_failure_chain(f"Execution exception: {str(e)}", attempt_id, inherited_chain_id)
                
                # Replenish to maintain concurrency (not exceeding target)
                while len(futures) < effective_workers and attempts < attempt_budget and len(successful_nodes) < target_count:
                    if not submit_one(executor):
                        break

            # Cancel remaining tasks if target reached
            if len(successful_nodes) >= target_count:
                for f in list(futures):
                    try:
                        f.cancel()
                    except Exception:
                        pass
                futures.clear()
        end_time = time.time()
        self.parallel_stats['parallel_calls'] += 1
        self.logger.info(
            f"Parallel batch expansion completed: generated {len(successful_nodes)} nodes in {end_time - start_time:.2f}s"
        )
        return successful_nodes[:target_count]

    def _generate_and_evaluate_parallel(self, task: Dict[str, Any]) -> Optional[MCTSNode]:
        parent_node = task['parent_node']
        image_path = task['image_path']
        attempt_id = task['attempt_id']
        failure_feedback = task.get('failure_feedback')
        
        try:
            cache_key = None
            if self.enable_caching:
                cache_key = f"{parent_node.node_id}_{attempt_id}"
                with self._lock:
                    if cache_key in self.qa_cache:
                        self.parallel_stats['cache_hits'] += 1
                        cached_result = self.qa_cache[cache_key]
                        if cached_result:
                            return self._create_node_from_cache(cached_result, parent_node)

            with self._client_semaphore:
                qa_data = self.qa_generator.generate_qa_pair(
                    parent_node=parent_node,
                    image_path=image_path,
                    failure_feedback=failure_feedback,
                )
                if not qa_data:
                    # Create failed node to pass failure reason
                    failed_node = MCTSNode("", "", parent_node.level + 1)
                    failed_node.failure_reason = "QA generation failed"
                    return failed_node

                temp_node = MCTSNode(
                    question=qa_data['question'],
                    answer=qa_data['answer'],
                    level=parent_node.level + 1,
                    generation_context={
                        'generated_from': parent_node.node_id,
                        'parallel_attempt': attempt_id,
                        'reasoning': qa_data.get('reasoning', ''),
                        'used_failure_feedback': failure_feedback is not None,
                    },
                )

                eval_cache_key = None
                if self.enable_caching:
                    hash_input = f"{parent_node.node_id}|{temp_node.question}|{temp_node.answer}".lower()
                    eval_cache_key = f"eval_{hashlib.sha1(hash_input.encode('utf-8')).hexdigest()}"
                    with self._lock:
                        quality_score = self.evaluation_cache.get(eval_cache_key)
                    
                    if quality_score is None:
                        quality_score, reasoning = self.qa_evaluator.evaluate_qa_quality(
                            child_node=temp_node,
                            parent_node=parent_node,
                            image_path=image_path,
                        )
                        if self.enable_caching:
                            with self._lock:
                                self.evaluation_cache[eval_cache_key] = (quality_score, reasoning)
                    else:
                        # Compatible with old cache format
                        if isinstance(quality_score, tuple):
                            quality_score, reasoning = quality_score
                        else:
                            reasoning = ""
                else:
                    quality_score, reasoning = self.qa_evaluator.evaluate_qa_quality(
                        child_node=temp_node,
                        parent_node=parent_node,
                        image_path=image_path,
                    )

            if quality_score >= self.batch_quality_threshold:
                temp_node.total_reward = quality_score
                temp_node.visit_count = 1
                if self.enable_caching and cache_key:
                    with self._lock:
                        self.qa_cache[cache_key] = {'qa_data': qa_data, 'quality_score': quality_score, 'reasoning': reasoning}
                with self._lock:
                    self.stats['successful_generations'] += 1
                return temp_node
            else:
                with self._lock:
                    self.stats['quality_rejections'] += 1
                # Log specific rejection reason for debug
                if reasoning:
                    self.logger.debug(f"Quality rejection (attempt {attempt_id}, score: {quality_score:.2f}): {reasoning[:100]}")
                # Create failed node to pass failure reason
                failed_node = MCTSNode("", "", parent_node.level + 1)
                failed_node.failure_reason = reasoning
                return failed_node
        except Exception as e:
            self.logger.warning(f"Parallel generation task failed (attempt {attempt_id}): {e}")
            # Create failed node to pass exception info
            failed_node = MCTSNode("", "", parent_node.level + 1)
            failed_node.failure_reason = f"Execution exception: {str(e)}"
            return failed_node

    def _create_node_from_cache(self, cached_result: Dict[str, Any], parent_node: MCTSNode) -> MCTSNode:
        qa_data = cached_result['qa_data']
        quality_score = cached_result['quality_score']
        node = MCTSNode(
            question=qa_data['question'],
            answer=qa_data['answer'],
            level=parent_node.level + 1,
            generation_context={'generated_from': parent_node.node_id, 'from_cache': True},
        )
        node.total_reward = quality_score
        node.visit_count = 1
        return node

    def _check_diversity_parallel(self, candidate: MCTSNode, existing_nodes: List[MCTSNode]) -> bool:
        # Configurable diversity check scope: full scan (all siblings) or recent window
        diversity_cfg = (self.config or {}).get('tree', {}).get('diversity', {})
        if bool(diversity_cfg.get('full_scan', False)):
            nodes_to_check = existing_nodes
        else:
            recent_n = int(diversity_cfg.get('recent_check_count', 5))
            check_count = min(len(existing_nodes), max(1, recent_n))
            nodes_to_check = existing_nodes[-check_count:]
        return self.diversity_checker.check_sibling_diversity(candidate, nodes_to_check)

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "total_generations": self.stats.get("total_generations", 0),
            "successful_generations": self.stats.get("successful_generations", 0),
            "failed_generations": self.stats.get("failed_generations", 0),
            "cache_hits": self.stats.get("cache_hits", 0),
            "diversity_rejections": self.stats.get("diversity_rejections", 0),
        }

    def get_parallel_statistics(self) -> Dict[str, Any]:
        base_stats = self.get_generation_statistics()
        base_stats.update({
            'parallel_stats': self.parallel_stats,
            'cache_enabled': self.enable_caching,
            'max_workers': self.max_workers,
        })
        return base_stats

    def clear_cache(self):
        if self.qa_cache:
            self.qa_cache.clear()
        if self.evaluation_cache:
            self.evaluation_cache.clear()
