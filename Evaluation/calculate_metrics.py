import json
import argparse
from collections import defaultdict


def calculate_metrics(predictions):
    """Calculate evaluation metrics."""
    # Filter out None predictions
    valid_predictions = [p for p in predictions if p is not None]
    total_items = len(valid_predictions)
    
    if total_items == 0:
        return {
            "total_items_evaluated": 0,
            "overall_correct_items": 0,
            "overall_accuracy": 0.0,
            "level_accuracies": {},
            "level_null_response_counts": {},
            "error_breakdown": {
                "total_failed_items": 0,
                "error_attribution_counts": {},
                "error_attribution_percentages": {}
            }
        }
    
    overall_correct_items = 0
    level_correct_counts = defaultdict(int)
    level_total_counts = defaultdict(int)
    level_null_response_counts = defaultdict(int)
    error_attribution_counts = defaultdict(int)

    for item in valid_predictions:
        all_correct = True
        first_error_level = None
        turns = item.get("conversation_turns", [])

        for turn in turns:
            level = turn["level"]
            is_correct = turn["is_correct"]
            model_response_key = turn.get("model_response_key")
            
            level_total_counts[level] += 1
            if model_response_key is None:
                level_null_response_counts[level] += 1

            if is_correct:
                level_correct_counts[level] += 1
            else:
                all_correct = False
                if first_error_level is None:
                    first_error_level = level
        
        if all_correct:
            overall_correct_items += 1
        elif first_error_level is not None:
            error_attribution_counts[first_error_level] += 1

    failed_items = total_items - overall_correct_items
    level_accuracies = {
        f"level_{level}_accuracy": (level_correct_counts[level] / total) * 100 if total > 0 else 0 
        for level, total in sorted(level_total_counts.items())
    }
    level_null_counts = {
        f"level_{level}_null_responses": level_null_response_counts[level] 
        for level in sorted(level_total_counts.keys())
    }

    error_breakdown = {
        "total_failed_items": failed_items,
        "error_attribution_counts": dict(sorted(error_attribution_counts.items())),
        "error_attribution_percentages": {
            f"level_{level}_caused_failure": (count / failed_items) * 100 if failed_items > 0 else 0 
            for level, count in sorted(error_attribution_counts.items())
        }
    }

    return {
        "total_items_evaluated": total_items,
        "overall_correct_items": overall_correct_items,
        "overall_accuracy": (overall_correct_items / total_items) * 100 if total_items > 0 else 0,
        "level_accuracies": level_accuracies,
        "level_null_response_counts": level_null_counts,
        "error_breakdown": error_breakdown
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from evaluation output.")
    parser.add_argument("input_file", type=str, help="Path to evaluation JSON file")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: 
            data = json.load(f)
    except FileNotFoundError: 
        return print(f"Error: Input file not found at {args.input_file}")
    except json.JSONDecodeError: 
        return print(f"Error: Could not decode JSON from {args.input_file}")

    predictions = data.get("predictions", [])
    if not predictions: 
        return print("No predictions found.")

    metrics = calculate_metrics(predictions)

    print("\n--- Metrics Report ---")
    print(f"Model: {data.get('model_name', 'N/A')}")
    print(f"Type: {data.get('evaluation_type', 'N/A')}")
    print("-" * 40)
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    print("\n--- End ---")


if __name__ == "__main__":
    main()
