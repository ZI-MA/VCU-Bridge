"""
Batch statistics and reporting tools
"""

from typing import Dict, Any, List


def create_final_results_summary(
    total_stats: Dict[str, Any],
    raw_config: Dict[str, Any],
    valid_image_paths: List[str]
) -> Dict[str, Any]:
    """Create final result summary"""
    # Derived aggregates
    per_image_times = total_stats.get('per_image_times', [])
    avg_time = (sum(per_image_times) / len(per_image_times)) if per_image_times else 0.0
    min_time = min(per_image_times) if per_image_times else 0.0
    max_time = max(per_image_times) if per_image_times else 0.0
    bf_avg = (total_stats.get('branching_factor_sum', 0.0) / max(1, total_stats.get('branching_factor_count', 0)))
    balance_avg = (total_stats.get('balance_score_sum', 0.0) / max(1, total_stats.get('balance_score_count', 0)))

    return {
        'execution_summary': {
            'total_execution_time_seconds': total_stats['total_execution_time'],
            'iterations_completed': total_stats['total_iterations'],
            'max_iterations': (raw_config.get("mcts", {}) or {}).get("max_iterations", 1000) * len(valid_image_paths),
            'total_api_calls': total_stats['total_api_calls'],
            'per_image_time_seconds': {
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
            }
        },
        'tree_statistics': {
            'total_nodes_final': total_stats['total_nodes'],
            'nodes_pruned': total_stats.get('nodes_pruned', 0),
            'successful_expansions': total_stats.get('successful_expansions', 0),
            'failed_generations': total_stats.get('failed_generations', 0),
            'failed_evaluations': total_stats.get('failed_evaluations', 0),
        },
        'quality_metrics': {
            # Consistent with single image report: success / (success + generation_failure)
            'expansion_success_rate': (
                (total_stats.get('successful_expansions', 0))
                / max(1, (total_stats.get('successful_expansions', 0) + total_stats.get('failed_generations', 0)))
            ),
            # Aggregated evaluation success rate: (success + gen_fail) / (success + gen_fail + eval_fail)
            'evaluation_success_rate': (
                (total_stats.get('successful_expansions', 0) + total_stats.get('failed_generations', 0))
                / max(
                    1,
                    total_stats.get('successful_expansions', 0)
                    + total_stats.get('failed_generations', 0)
                    + total_stats.get('failed_evaluations', 0),
                )
            ),
        },
        'sharegpt_conversion': {
            'total_conversations': total_stats['total_conversations'],
            'total_questions': total_stats['total_questions'],
        },
        'batch_processing': {
            'images_processed': total_stats['images_processed'],
            'images_total': len(valid_image_paths),
        },
        'structure_summary': {
            'avg_branching_factor': bf_avg,
            'avg_balance_score': balance_avg,
        },
        'paths_summary': {
            'total_paths': total_stats.get('paths_total', 0),
            'avg_path_length': (total_stats.get('avg_path_length_sum', 0.0) / max(1, total_stats.get('images_processed', 0))),
            'max_path_length': total_stats.get('max_path_length_max', 0),
        }
    }


def print_final_statistics(
    results: Dict[str, Any],
    failed_images: List[str],
    raw_config: Dict[str, Any],
    output_path: str,
    tree_output_path: str = None
):
    """Print final statistics"""
    
    exec_summary = results['execution_summary']
    print(f"Total Execution Time: {exec_summary['total_execution_time_seconds']:.2f}s")
    print(f"Total Iterations: {exec_summary['iterations_completed']}/{exec_summary['max_iterations']}")
    print(f"Total API Calls: {exec_summary['total_api_calls']}")
    pit = exec_summary.get('per_image_time_seconds', {})
    if pit:
        print(f"Per-Image Time: avg {pit.get('avg', 0):.2f}s, min {pit.get('min', 0):.2f}s, max {pit.get('max', 0):.2f}s")
    
    batch_info = results['batch_processing']
    print(f"Images Processed: {batch_info['images_processed']}/{batch_info['images_total']}")
    print(f"Images Failed: {len(failed_images)}/{batch_info['images_total']}")
    
    # Show parallel config info
    parallel_cfg = raw_config.get('parallel', {}) or {}
    node_workers = int(parallel_cfg.get('nodes', 1))
    batch_workers = int(parallel_cfg.get('batchs', 1))
    image_workers = int(parallel_cfg.get('images', 1))
    print(f"Image Parallel: {image_workers} workers ({'serial' if image_workers == 1 else 'parallel'})")
    print(f"Node Parallel: {node_workers} workers ({'serial' if node_workers == 1 else 'parallel'})")
    print(f"Batch Parallel: {batch_workers} workers ({'serial' if batch_workers == 1 else 'parallel'})")
    
    tree_stats = results['tree_statistics']
    print(f"Total Nodes Generated: {tree_stats['total_nodes_final']}")
    if 'nodes_pruned' in tree_stats:
        print(f"Nodes Pruned: {tree_stats['nodes_pruned']}")
    if 'successful_expansions' in tree_stats:
        print(f"Expansions: success {tree_stats['successful_expansions']}, gen_fail {tree_stats['failed_generations']}, eval_fail {tree_stats['failed_evaluations']}")
    print(f"Success Rate: {results['quality_metrics']['expansion_success_rate']:.2%}")
    
    conv_stats = results['sharegpt_conversion']
    print(f"Total Conversations Generated: {conv_stats.get('total_conversations', 0)}")
    print(f"Total Questions: {conv_stats.get('total_questions', 0)}")

    # Optional structure & path summaries
    ss = results.get('structure_summary')
    if ss:
        print(f"Avg Branching Factor: {ss.get('avg_branching_factor', 0):.2f}")
        print(f"Avg Balance Score: {ss.get('avg_balance_score', 0):.2f}")
    ps = results.get('paths_summary')
    if ps:
        print(f"Paths: total {ps.get('total_paths', 0)}, avg_len {ps.get('avg_path_length', 0):.2f}, max_len {ps.get('max_path_length', 0)}")
    print(f"\nCombined output saved to: {output_path}")
    
    if tree_output_path:
        print(f"Combined tree states saved to: {tree_output_path}")
