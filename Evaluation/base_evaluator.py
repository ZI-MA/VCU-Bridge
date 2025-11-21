import os
import json
import base64
import asyncio
from abc import ABC
import openai
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import random
from Evaluation.answer_parser import parse_multiple_choice_answer


class RetryableError(Exception):
    """Retryable error (network, service)"""
    pass

class ParseError(Exception):
    """Parse error (format)"""
    pass


class BaseEvaluator(ABC):
    """Base evaluator for vision models on hierarchical QA."""
    
    MAX_NETWORK_RETRIES = 3
    
    def __init__(self, input_data_path, image_dir, results_dir, output_filename, 
                 context_mode=False, max_delay=30.0, temperature=0.0):
        self.input_data_path = input_data_path
        self.image_dir = image_dir
        self.results_dir = results_dir
        self.output_filename = output_filename
        self.model_name = "unknown"
        self.context_mode = context_mode
        self.max_delay = max_delay
        self.temperature = temperature
        self.client: openai.AsyncOpenAI = None
        self.none_response_counts = {}
        self.system_prompt = "You are a helpful assistant that answers multiple-choice questions based on images. Respond with ONLY the single letter (A, B, C, or D) of the best option. Do not include any explanation or additional text."

    def encode_image(self, image_path):
        """Encode image to base64."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def get_image_path(self, image_id):
        """Find image path by ID."""
        for ext in ['.jpg', '.png', '.webp', '.jpeg']:
            path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                return path
        return None

    def _calculate_backoff_time(self, attempt):
        """Calculate exponential backoff delay."""
        exponential_delay = 1.0 * (2 ** attempt)
        return min(exponential_delay, self.max_delay)

    async def call_vision_model(self, messages):
        """Call vision model API with error handling."""
        if not self.client:
            raise ParseError("API client not initialized.")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            message = response.choices[0].message
            if message is None or message.content is None:
                raise RetryableError("API returned None")
            
            if not message.content.strip():
                raise ParseError("Empty content")
            
            parsed = parse_multiple_choice_answer(message.content)
            if parsed is None:
                raise ParseError(message.content[:100])
            
            return parsed
            
        except openai.APITimeoutError as e:
            raise RetryableError(f"Timeout: {e}")
        except openai.APIConnectionError as e:
            raise RetryableError(f"Connection: {e}")
        except openai.RateLimitError as e:
            raise RetryableError(f"Rate limit: {e}")
        except openai.APIStatusError as e:
            if e.status_code in [500, 502, 503, 504, 524, 400]:
                raise RetryableError(f"Server {e.status_code}: {e}")
            else:
                raise ParseError(f"API {e.status_code}: {e}")
        except ValueError as e:
            raise ParseError(str(e))
        except (RetryableError, ParseError):
            raise
        except Exception as e:
            raise ParseError(str(e))

    async def _process_item(self, item):
        """Process single item."""
        item_id = item['id']
        image_path = self.get_image_path(item_id)
        if not image_path: 
            return None
        encoded_image = self.encode_image(image_path)
        if not encoded_image: 
            return None

        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'

        prediction_result = {"id": item_id, "image_path": image_path, "conversation_turns": []}

        for level_index, question_answer_level in enumerate(item['qa_levels']):
            question = question_answer_level['question']
            options = question_answer_level['options']
            prompt_text = f"{question}\nOptions:\n" + "\n".join([f"{option['option_key']}. {option['option_text']}" for option in options])

            messages = [{"role": "system", "content": self.system_prompt}]
            
            if self.context_mode and level_index > 0:
                for prev_turn in prediction_result["conversation_turns"]:
                    prev_text = f"{prev_turn['question']}\nOptions:\n" + "\n".join([f"{k}. {v}" for k, v in prev_turn['options'].items()])
                    messages.append({"role": "user", "content": prev_text})
                    messages.append({"role": "assistant", "content": prev_turn['model_response_key']})
                prompt_text = f"The above Q&As provide context. Now carefully consider:\n{prompt_text}"
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                ]
            })

            model_response_key = None
            last_error = None

            for network_retry in range(self.MAX_NETWORK_RETRIES):
                try:
                    model_response_key = await self.call_vision_model(messages=messages)
                    break
                except RetryableError as e:
                    last_error = e
                    if network_retry < self.MAX_NETWORK_RETRIES - 1:
                        await asyncio.sleep(self._calculate_backoff_time(network_retry))
                except ParseError as e:
                    last_error = e
                    break

            if model_response_key is None:
                level = question_answer_level['level']
                self.none_response_counts[level] = self.none_response_counts.get(level, 0) + 1

            correct_answer_key = question_answer_level['correct_answer_key'].strip().upper()
            prediction_result["conversation_turns"].append({
                "level": question_answer_level['level'],
                "question": question,
                "options": {option['option_key']: option['option_text'] for option in options},
                "correct_answer_key": correct_answer_key,
                "model_response_key": model_response_key,
                "is_correct": model_response_key == correct_answer_key if model_response_key else False,
                "ground_truth_text": next((option['option_text'] for option in options if option['option_key'] == correct_answer_key), "Unknown"),
                "predicted_text": next((option['option_text'] for option in options if option['option_key'] == model_response_key), "Error") if model_response_key else "Error"
            })

        return prediction_result

    async def run_evaluation(self, parallel_workers=1):
        """Run evaluation."""
        if hasattr(self, 'test_connection'):
            await self.test_connection()

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        output_path = os.path.join(self.results_dir, self.output_filename)
        
        try:
            with open(self.input_data_path, 'r', encoding='utf-8') as f:
                generated_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Generated data file not found at {self.input_data_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.input_data_path}")
            return

        existing_results = None
        items_to_retry = []
        retry_mode = False
        
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                
                null_item_ids = set()
                predictions = existing_results.get('predictions', [])
                valid_ids = set()
                
                # First pass: check existing valid predictions for partial failures
                for prediction in predictions:
                    if prediction is not None:
                        item_id = prediction.get('id')
                        if item_id:
                            valid_ids.add(item_id)
                            has_null = any(turn.get('model_response_key') is None 
                                          for turn in prediction.get('conversation_turns', []))
                            if has_null:
                                null_item_ids.add(item_id)
                
                # Second pass: identify missing or None items
                for item in generated_data:
                    if item['id'] not in valid_ids:
                        null_item_ids.add(item['id'])
                
                if not null_item_ids:
                    print(f"No null responses found. Skipping evaluation.")
                    return
                
                id_to_item = {item['id']: item for item in generated_data}
                items_to_retry = [id_to_item[item_id] for item_id in null_item_ids if item_id in id_to_item]
                
                if not items_to_retry:
                    return
                
                retry_mode = True
                
            except (json.JSONDecodeError, KeyError):
                existing_results = None
                retry_mode = False

        items_to_process = items_to_retry if retry_mode else generated_data
        total_items = len(items_to_process)
        all_predictions = []

        mode_str = "with_context" if self.context_mode else "independent"
        print(f"Evaluating {total_items} items with {self.model_name} in {mode_str} mode ({parallel_workers} workers)")

        if parallel_workers > 1:
            semaphore = asyncio.Semaphore(parallel_workers)
            
            async def process_with_semaphore(index, item):
                async with semaphore:
                    return (index, await self._process_item(item))
            
            tasks = [process_with_semaphore(i, item) for i, item in enumerate(items_to_process)]
            results_array = [None] * total_items
            
            pbar = async_tqdm(total=total_items, desc="Evaluating")
            
            for completed_task in asyncio.as_completed(tasks):
                try:
                    index, result = await completed_task
                    results_array[index] = result
                except Exception as error:
                    print(f"Error: {error}")
                finally:
                    pbar.update(1)
            
            pbar.close()
            all_predictions = results_array
        else:
            for item in tqdm(items_to_process, total=total_items, desc="Evaluating"):
                result = await self._process_item(item)
                all_predictions.append(result)

        if retry_mode and existing_results:
            retry_results_map = {prediction['id']: prediction for prediction in all_predictions if prediction is not None}
            
            merged_predictions = []
            for prediction in existing_results.get('predictions', []):
                if prediction is None:
                    merged_predictions.append(None)
                elif prediction.get('id') in retry_results_map:
                    merged_predictions.append(retry_results_map[prediction['id']])
                else:
                    merged_predictions.append(prediction)
            
            all_predictions = merged_predictions

        results = {
            "model_name": self.model_name,
            "evaluation_type": "sequential_with_context" if self.context_mode else "independent_level",
            "total_items_processed": len([prediction for prediction in all_predictions if prediction is not None]),
            "predictions": all_predictions
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"\nResults saved to {output_path}")
        
        if self.none_response_counts:
            print("None responses by level:", dict(sorted(self.none_response_counts.items())))


class OpenAIEvaluator(BaseEvaluator):
    """Evaluator for OpenAI-compatible APIs."""
    
    def __init__(self, input_data_path, image_dir, results_dir, output_filename,
                 model_name="gpt-4o", context_mode=False,
                 api_key=None, base_url=None, max_delay=30.0, temperature=0.0):
        super().__init__(input_data_path, image_dir, results_dir, output_filename, 
                        context_mode, max_delay, temperature)
        self.model_name = model_name

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY.")
        
        base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def test_connection(self):
        """Test API connection before starting evaluation."""
        try:
            print(f"Testing connection to {self.client.base_url}...")
            await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            print("API connection test passed.")
        except Exception as e:
            print(f"\n[Connection Test Failed]")
            print(f"Error: {e}")
            print("Check your PROXY and API KEY settings.\n")
            raise e


class LocalEvaluator(BaseEvaluator):
    """
    Evaluator for local models via OpenAI-compatible API (e.g., vLLM).
    
    Example: Qwen3-VL models
    """
    
    def __init__(self, input_data_path, image_dir, results_dir, output_filename, 
                 model_name="Qwen/Qwen3-VL-8B-Instruct", context_mode=False, 
                 max_delay=30.0, temperature=0.0, base_url=None):
        super().__init__(input_data_path, image_dir, results_dir, output_filename, 
                        context_mode, max_delay, temperature)
        self.model_name = model_name
        
        if base_url is None:
            base_url = "http://localhost:8000/v1"
        
        self.client = openai.AsyncOpenAI(
            api_key="EMPTY",
            base_url=base_url
        )
