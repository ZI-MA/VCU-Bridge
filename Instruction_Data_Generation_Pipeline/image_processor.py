import os
import json
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .utils.orchestrator import MCTSInstructionDataGenerator


def parse_image_paths(image_path: Any) -> Tuple[List[str], Path]:
    """Parse image path configuration, returns list of paths and failed images file path."""
    image_paths = []
    failed_images_file = None
    
    if isinstance(image_path, list):
        image_paths = image_path
    elif image_path.endswith('.json'):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path JSON file not found: {image_path}")
        with open(image_path, 'r') as f:
            image_paths = json.load(f)
        
        json_path = Path(image_path)
        failed_images_file = json_path.parent / f"{json_path.stem}_failed{json_path.suffix}"
    else:
        image_paths = [image_path]
    
    return image_paths, failed_images_file


def validate_image_paths(image_paths: List[str], logger: logging.Logger = None) -> List[str]:
    """Validate image paths exist."""
    if not logger:
        logger = logging.getLogger(__name__)
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            logger.warning(f"Image not found (skipping): {img_path}")
            
    valid_image_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_image_paths:
        raise ValueError("No valid image paths found")
    
    return valid_image_paths


def process_single_image(
    img_path: str,
    img_index: int,
    total_images: int,
    config_dict: Dict[str, Any],
    shared_client,
    shared_semaphore,
    load_tree_path: str = None,
    logger=None
) -> Tuple[str, Dict[str, Any], bool]:
    """Process single image using MCTS-driven instruction data generation."""
    if not logger:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Processing image {img_index+1}/{total_images}: {img_path}")
        
        generator = MCTSInstructionDataGenerator(
            client=shared_client,
            config=config_dict,
            shared_semaphore=shared_semaphore,
        )

        if load_tree_path:
            logger.info(f"Loading existing tree from {load_tree_path}")
            generator.load_existing_tree(load_tree_path)
        
        results = generator.generate_instruction_data(
            image_path=img_path,
            output_path=None,
            tree_save_path=None,
            top_k_paths=None
        )

        # Check if generation failed
        try:
            if not results.get('sharegpt_data'):
                return img_path, {**results, 'error': 'no conversations generated'}, True
            ts = results.get('tree_statistics', {}) or {}
            if int(ts.get('successful_expansions', 0)) <= 0:
                return img_path, {**results, 'error': 'no successful expansions'}, True
            pe = results.get('path_extraction', {}) or {}
            if int(pe.get('total_paths_extracted', 0)) <= 0:
                return img_path, {**results, 'error': 'no paths extracted'}, True
        except Exception:
            pass

        return img_path, results, False
        
    except Exception as e:
        logger.error(f"Failed to process image {img_path}: {e}")
        return img_path, {'error': str(e)}, True

