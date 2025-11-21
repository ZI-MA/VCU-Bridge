"""
Thread-safe result collector
"""

import threading
from typing import List, Dict, Any


class ResultCollector:
    """Thread-safe result collector for aggregating multi-image processing results"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.all_sharegpt_data = []
        self.all_tree_states = []
        self.failed_images = []
        self.total_stats = {
            'total_execution_time': 0,
            'total_iterations': 0,
            'total_api_calls': 0,
            'total_nodes': 0,
            'total_conversations': 0,
            'total_questions': 0,
            'images_processed': 0,
            # MCTS quality and pruning
            'successful_expansions': 0,
            'failed_generations': 0,
            'failed_evaluations': 0,
            'nodes_pruned': 0,
            # Structure metrics (averaged if available)
            'branching_factor_sum': 0.0,
            'branching_factor_count': 0,
            'balance_score_sum': 0.0,
            'balance_score_count': 0,
            # Paths
            'paths_total': 0,
            'avg_path_length_sum': 0.0,
            'max_path_length_max': 0,
            # Time per image
            'per_image_times': [],
        }
    
    def add_result(self, img_path: str, results: Dict[str, Any], failed: bool = False):
        """Add processing result for a single image"""
        with self.lock:
            if failed:
                self.failed_images.append(img_path)
                return
                
            # Collect data
            if 'sharegpt_data' in results:
                self.all_sharegpt_data.extend(results['sharegpt_data'])
            
            if 'tree_state' in results:
                # Use image_path as key: overwrite if exists, otherwise append
                new_item = {
                    'image_path': img_path,
                    'tree_state': results['tree_state']
                }
                replaced = False
                for idx, item in enumerate(self.all_tree_states):
                    try:
                        if isinstance(item, dict) and item.get('image_path') == img_path:
                            self.all_tree_states[idx] = new_item
                            replaced = True
                            break
                    except Exception:
                        # Ignore exception, continue trying
                        continue
                if not replaced:
                    self.all_tree_states.append(new_item)
            
            # Accumulate statistics
            exec_summary = results.get('execution_summary', {})
            tree_stats = results.get('tree_statistics', {})
            conv_stats = results.get('sharegpt_conversion', {})
            
            self.total_stats['total_execution_time'] += exec_summary.get('total_execution_time_seconds', 0)
            self.total_stats['per_image_times'].append(exec_summary.get('total_execution_time_seconds', 0))
            self.total_stats['total_iterations'] += exec_summary.get('iterations_completed', 0)
            self.total_stats['total_api_calls'] += exec_summary.get('total_api_calls', 0)
            self.total_stats['total_nodes'] += tree_stats.get('total_nodes_final', 0)
            self.total_stats['successful_expansions'] += tree_stats.get('successful_expansions', 0)
            self.total_stats['failed_generations'] += tree_stats.get('failed_generations', 0)
            self.total_stats['failed_evaluations'] += tree_stats.get('failed_evaluations', 0)
            self.total_stats['nodes_pruned'] += tree_stats.get('nodes_pruned', 0)
            self.total_stats['total_conversations'] += conv_stats.get('total_conversations', 0)
            self.total_stats['total_questions'] += conv_stats.get('total_questions', 0)

            # Structure statistics (optional)
            st = results.get('structure_statistics', {})
            if isinstance(st, dict):
                bf = (st.get('branching_factor') or {}).get('average')
                if isinstance(bf, (int, float)):
                    self.total_stats['branching_factor_sum'] += float(bf)
                    self.total_stats['branching_factor_count'] += 1
                bs = (st.get('balance_info') or {}).get('balance_score')
                if isinstance(bs, (int, float)):
                    self.total_stats['balance_score_sum'] += float(bs)
                    self.total_stats['balance_score_count'] += 1

            # Paths (optional)
            p = results.get('path_extraction', {})
            if isinstance(p, dict):
                self.total_stats['paths_total'] += p.get('total_paths_extracted', 0)
                self.total_stats['avg_path_length_sum'] += p.get('average_path_length', 0.0)
                self.total_stats['max_path_length_max'] = max(
                    self.total_stats['max_path_length_max'], p.get('max_path_length', 0)
                )
            self.total_stats['images_processed'] += 1
