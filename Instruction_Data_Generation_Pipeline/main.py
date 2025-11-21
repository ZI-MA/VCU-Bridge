"""CLI entry for MCTS-driven hierarchical reasoning data generation."""

import argparse
import json
import os
import sys
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from .config import RuntimeConfig, setup_logging
from .utils.services.clients import create_client
from .utils.random import set_global_seed
from .utils.batch import (
    ResultCollector,
    create_final_results_summary,
    print_final_statistics
)
from .image_processor import (
    process_single_image,
    parse_image_paths,
    validate_image_paths
)


def _handle_image_result(img_index, img_path, results, failed, total_images, 
                        result_collector, output_path, tree_output_path, logger):
    """Handle single image result."""
    if failed:
        result_collector.add_result(img_path, results, failed=True)
    else:
        exec_summary = results.get('execution_summary', {})
        conv_stats = results.get('sharegpt_conversion', {})
        result_collector.add_result(img_path, results, failed=False)

        # Incremental save (JSONL format)
        with result_collector.lock:
            new_conversations = results.get('sharegpt_data', [])
            if new_conversations:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for conv in new_conversations:
                        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

            if tree_output_path:
                tree_item = {
                    'image_path': img_path,
                    'tree_state': results.get('tree_state', {})
                }
                with open(tree_output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(tree_item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="MCTS-driven Reasoning Path Discovery")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    try:
        cfg = RuntimeConfig(args.config)
        config_dict = cfg.to_dict()

        log_cfg = config_dict.get("logging") or {}
        setup_logging({
            "log_level": log_cfg.get("log_level", config_dict.get("log_level", "INFO")),
            "log_file": log_cfg.get("log_file", config_dict.get("log_file")),
        })
        logger = logging.getLogger(__name__)

        # Set random seed
        run_cfg = config_dict.get('run', {})
        seed = run_cfg.get('random_seed')
        set_global_seed(seed)

        # Parse I/O config
        io_cfg = config_dict.get('io', {})
        image_path = io_cfg.get('image_path')
        output_path = io_cfg.get('output')
        tree_output_path = io_cfg.get('tree_output')
        load_tree_path = io_cfg.get('load_tree')
        
        if not image_path or not output_path:
            raise ValueError("Config must include 'image_path' and 'output'.")
        
        # Parse and validate
        image_paths, failed_images_file = parse_image_paths(image_path)
        valid_image_paths = validate_image_paths(image_paths, logger)
        
        logger.info(f"Found {len(valid_image_paths)} valid images")
        
        # Resume mechanism: check completed images
        completed_images = set()
        if tree_output_path and os.path.exists(tree_output_path):
            logger.info(f"Detected existing tree output: {tree_output_path}")
            with open(tree_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict) and 'image_path' in item and item['tree_state']['metadata']['total_expansions'] > 0:
                            completed_images.add(item['image_path'])
            logger.info(f"Found {len(completed_images)} completed images")
        
        # Filter out completed images
        images_to_process = [img for img in valid_image_paths if img not in completed_images]
        
        if len(completed_images) > 0:
            logger.info(f"Resume mode: {len(completed_images)} completed, {len(images_to_process)} remaining")
        
        if not images_to_process:
            logger.info("All images already processed!")
            sys.exit(0)
        
        valid_image_paths = images_to_process
        
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        logger.info("Initializing model client...")
        client = create_client(config_dict)
        
        # Test connection
        if hasattr(client, 'test_connection'):
            try:
                client.test_connection()
            except Exception:
                sys.exit(1)
        
        # Parallel config
        parallel_cfg = config_dict.get('parallel', {})
        image_parallel_workers = int(parallel_cfg.get('images', 1))
        client_cfg = config_dict.get('client', {})
        client_max_concurrency = int(client_cfg.get('max_concurrency', 10))
        
        # Global API concurrency control
        global_api_semaphore = threading.BoundedSemaphore(client_max_concurrency)
        
        # Result collector
        result_collector = ResultCollector()
        
        # Load existing results
        if os.path.exists(output_path):
            logger.info(f"Loading existing output: {output_path}")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result_collector.all_sharegpt_data.append(json.loads(line))
            logger.info(f"Loaded {len(result_collector.all_sharegpt_data)} conversations")

        if tree_output_path and os.path.exists(tree_output_path):
            with open(tree_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result_collector.all_tree_states.append(json.loads(line))
            logger.info(f"Loaded {len(result_collector.all_tree_states)} tree states")

        logger.info("Starting MCTS-driven reasoning path discovery...")
        logger.info(f"Image workers: {image_parallel_workers}, API concurrency: {client_max_concurrency}")
        
        with ThreadPoolExecutor(max_workers=image_parallel_workers) as executor:
            future_to_image = {}
            for i, img_path in enumerate(valid_image_paths):
                future = executor.submit(
                    process_single_image,
                    img_path, i, len(valid_image_paths),
                    config_dict, client, global_api_semaphore, load_tree_path, logger
                )
                future_to_image[future] = (i, img_path)
            
            with tqdm(total=len(valid_image_paths), desc="Processing", unit="img") as pbar:
                for future in as_completed(future_to_image):
                    i, img_path = future_to_image[future]
                    
                    try:
                        img_path_result, results, failed = future.result()
                        _handle_image_result(i, img_path, results, failed, len(valid_image_paths), 
                                            result_collector, output_path, tree_output_path, logger)
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Exception processing {img_path}: {e}")
                        result_collector.add_result(img_path, {'error': str(e)}, failed=True)
                        pbar.update(1)

        # Collect results
        all_sharegpt_data = result_collector.all_sharegpt_data
        all_tree_states = result_collector.all_tree_states
        failed_images = result_collector.failed_images
        total_stats = result_collector.total_stats
        
        # Save failed images
        if failed_images and failed_images_file:
            logger.info(f"Saving {len(failed_images)} failed images to {failed_images_file}")
            with open(failed_images_file, 'w', encoding='utf-8') as f:
                json.dump(failed_images, f, ensure_ascii=False, indent=2)
        
        # Final statistics
        results = create_final_results_summary(total_stats, config_dict, valid_image_paths)
        print_final_statistics(results, failed_images, config_dict, output_path, tree_output_path)

        logger.info("Batch execution completed successfully")
    except Exception as e:
        logging.getLogger(__name__).error(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

