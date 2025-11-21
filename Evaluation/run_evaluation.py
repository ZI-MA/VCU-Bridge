import argparse
import os
import asyncio
from Evaluation.base_evaluator import OpenAIEvaluator, LocalEvaluator
from dotenv import load_dotenv

load_dotenv()


def get_default_paths():
    """Get default paths from environment variables."""
    return {
        'data_dir': os.getenv('DATA_DIR', 'Data'),
        'results_dir': os.getenv('RESULTS_DIR', 'Result'),
        'image_dir': os.getenv('IMAGE_DIR', 'Data/Image'),
    }


def infer_benchmark_name(input_filename):
    """Infer benchmark name from input filename."""
    name_without_ext = os.path.splitext(os.path.basename(input_filename))[0]
    
    known_benchmarks = [
        'Aesthetic-Appreciation',
        'Affective-Reasoning',
        'Implication-Understanding'
    ]
    
    for bench in known_benchmarks:
        if name_without_ext == bench:
            return bench
    
    return name_without_ext


def get_image_dir_for_benchmark(benchmark_name, data_dir='Data'):
    """Get image directory path for a benchmark."""
    default_image_base = os.path.join(data_dir, 'Image')
    image_dir = os.path.join(default_image_base, benchmark_name)
    
    if os.path.exists(image_dir):
        return image_dir
    
    env_image_dir = os.getenv('IMAGE_DIR')
    if env_image_dir and os.path.exists(env_image_dir):
        return env_image_dir
    
    return image_dir


def main():
    defaults = get_default_paths()
    
    parser = argparse.ArgumentParser(
        description="Run evaluation for vision models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with OpenAI API (default model: gpt-4o)
  python -m Evaluation.run_evaluation openai --input Implication-Understanding.json
  
  # Evaluate with different OpenAI model
  python -m Evaluation.run_evaluation openai --input Implication-Understanding.json --model gpt-4o-mini
  
  # Evaluate with local model in context mode (default: Qwen/Qwen3-VL-8B-Instruct)
  python -m Evaluation.run_evaluation local --input Implication-Understanding.json --context_mode
  
  # Evaluate with custom parallel workers for faster processing
  python -m Evaluation.run_evaluation openai --input Implication-Understanding.json --parallel 4
        """
    )
    
    parser.add_argument("evaluator", choices=["openai", "local"], 
                       help="Evaluator type: openai or local (vLLM)")
    parser.add_argument("--input", type=str, required=True,
                       help="Input filename (in data directory)")
    
    parser.add_argument("--model", type=str, 
                       help="Model name (default: gpt-4o for openai, Qwen/Qwen3-VL-8B-Instruct for local)")
    parser.add_argument("--parallel", type=int, default=1, 
                       help="Number of parallel workers (default: 1)")
    parser.add_argument("--context_mode", action="store_true", 
                       help="Enable context mode (use previous Q&A as context)")
    parser.add_argument("--temperature", type=float, default=0.0, 
                       help="Sampling temperature (default: 0.0)")
    parser.add_argument("--base_url", type=str, 
                       help="Custom API base URL (for OpenAI-compatible APIs)")

    args = parser.parse_args()

    if args.evaluator == "openai":
        model_name = args.model or "gpt-4o"
    else:
        model_name = args.model or "Qwen/Qwen3-VL-8B-Instruct"
    
    output_filename = model_name.split('/')[-1].replace('-', '_') + ".json"
    
    input_filename_without_ext = os.path.splitext(os.path.basename(args.input))[0]
    mode_subdir = "Context" if args.context_mode else "Indep"
    results_dir = os.path.join(defaults['results_dir'], input_filename_without_ext, mode_subdir)
    os.makedirs(results_dir, exist_ok=True)
    
    input_data_path = os.path.join(defaults['data_dir'], args.input)
    
    benchmark_name = infer_benchmark_name(args.input)
    env_image_dir = os.getenv('IMAGE_DIR')
    if env_image_dir:
        subdir_path = os.path.join(env_image_dir, benchmark_name)
        if os.path.exists(subdir_path):
            image_dir = subdir_path
        else:
            image_dir = env_image_dir
    else:
        image_dir = get_image_dir_for_benchmark(benchmark_name, defaults['data_dir'])
    
    print(f"Benchmark: {benchmark_name}")
    print(f"Image directory: {image_dir}")
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory does not exist: {image_dir}")
        print("Please ensure images are placed in the correct directory structure.")

    if args.evaluator == "openai":
        evaluator = OpenAIEvaluator(
            input_data_path=input_data_path,
            image_dir=image_dir,
            results_dir=results_dir,
            output_filename=output_filename,
            model_name=model_name,
            context_mode=args.context_mode,
            temperature=args.temperature,
            base_url=args.base_url
        )
    else:
        base_url = args.base_url or "http://localhost:8000/v1"
        evaluator = LocalEvaluator(
            input_data_path=input_data_path,
            image_dir=image_dir,
            results_dir=results_dir,
            output_filename=output_filename,
            model_name=model_name,
            context_mode=args.context_mode,
            temperature=args.temperature,
            base_url=base_url
        )

    asyncio.run(evaluator.run_evaluation(parallel_workers=args.parallel))


if __name__ == "__main__":
    main()
