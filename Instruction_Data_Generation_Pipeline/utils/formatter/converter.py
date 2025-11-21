"""
Hierarchical Reasoning Data - ShareGPT Converter
"""

import uuid
import random
from typing import List, Dict, Any, Optional
import logging

from ..nodes import MCTSNode
from ..random import derive_int_seed

logger = logging.getLogger(__name__)


class ShareGPTConverter:
    def __init__(self, seed: int = None, mode: str = "multiple_choice"):
        self.conversation_id_counter = 0
        self._seed = seed
        self.mode = mode  # "multiple_choice" or "qa"

    def convert_paths_to_sharegpt(self, paths: List[List[MCTSNode]], image_path: str = None, tree=None) -> List[Dict[str, Any]]:
        self.tree = tree
        sharegpt_conversations = []
        for path in paths:
            if len(path) == 0:
                continue
            conversation = self._convert_single_path(path, image_path)
            if conversation:
                sharegpt_conversations.append(conversation)
        return sharegpt_conversations

    def _convert_single_path(self, path: List[MCTSNode], image_path: str = None) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        non_root_nodes = [node for node in path if not node.is_root]
        if not non_root_nodes:
            return None
        self.conversation_id_counter += 1
        conversation_id = f"mcts_conv_{self.conversation_id_counter:06d}_{uuid.uuid4().hex[:8]}"
        conversation = {
            "id": conversation_id,
            "conversations": [],
            "source": "mcts_hierarchical_generator",
            "metadata": {
                "path_length": len(non_root_nodes),
                "original_path_length": len(path),
                "levels": [node.level for node in non_root_nodes],
                "total_visits": sum(node.visit_count for node in non_root_nodes),
                "avg_reward": sum(node.average_reward for node in non_root_nodes) / len(non_root_nodes) if non_root_nodes else 0.0,
                "conversion_mode": self.mode,
            },
        }
        if image_path:
            conversation["images"] = [image_path]
        for turn_index, node in enumerate(non_root_nodes):
            human_turn = self._create_human_turn(node, path, turn_index, image_path)
            if human_turn:
                conversation["conversations"].append(human_turn)
                gpt_turn = self._create_gpt_turn(node)
                conversation["conversations"].append(gpt_turn)
        return conversation

    def _create_human_turn(self, current_node: MCTSNode, full_path: List[MCTSNode], turn_index: int, image_path: str = None) -> Dict[str, Any]:
        content_parts = []
        if turn_index == 0 and image_path:
            content_parts.append("<image>")
        content_parts.append(current_node.question)
        
        if self.mode == "qa":
            # QA mode: only include the question
            full_content = "\n".join(content_parts)
        else:
            # Multiple choice mode: include options and instructions
            options = self._generate_multiple_choice_options(current_node, full_path)
            option_letters = ['A', 'B', 'C', 'D']
            for i, option in enumerate(options):
                letter = option_letters[i]
                content_parts.append(f"{letter}. {option['text']}")
            content_parts.append("\nPlease answer with the letter only.")
            full_content = "\n".join(content_parts)
        
        return {"from": "human", "value": full_content}

    def _create_gpt_turn(self, node: MCTSNode) -> Dict[str, Any]:
        if self.mode == "qa":
            # QA mode: return the full answer
            return {"from": "gpt", "value": node.answer}
        else:
            # Multiple choice mode: return the option letter
            correct_letter = node.generation_context.get('correct_answer_letter', 'A')
            return {"from": "gpt", "value": correct_letter}

    def _generate_multiple_choice_options(self, current_node: MCTSNode, full_path: List[MCTSNode]) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []
        options.append({'text': current_node.answer, 'is_correct': True})
        distractors = self._generate_distractors(current_node, self.tree, full_path)
        for distractor in distractors[:3]:
            options.append({'text': distractor, 'is_correct': False})
        while len(options) < 4:
            generic_distractor = self._generate_generic_distractor(current_node, options)
            options.append({'text': generic_distractor, 'is_correct': False})
        options = self._postprocess_options(options, current_node.level)
        try:
            rng_seed = derive_int_seed(current_node.node_id, base=(self._seed or 0))
            rnd = random.Random(rng_seed)
            rnd.shuffle(options)
        except Exception:
            random.shuffle(options)
        option_letters = ['A', 'B', 'C', 'D']
        for i, option in enumerate(options):
            if option['is_correct']:
                current_node.generation_context['correct_answer_letter'] = option_letters[i]
                break
        return options

    def _postprocess_options(self, options: List[Dict[str, Any]], level: int, max_len: int = 140) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        seen_lower = set()
        for opt in options:
            text = self._trim_option_text(str(opt['text']), max_len=max_len)
            text_norm = text.strip().lower()
            if not text_norm or text_norm in seen_lower:
                continue
            seen_lower.add(text_norm)
            processed.append({'text': text, 'is_correct': opt['is_correct']})
        while len(processed) < 4:
            filler = self._generate_generic_filler(level=level, existing_texts={p['text'].lower() for p in processed})
            processed.append({'text': filler, 'is_correct': False})
        return processed[:4]

    def _trim_option_text(self, text: str, max_len: int = 140) -> str:
        t = text.strip().replace('\n', ' ')
        period_idx = t.find('. ')
        if 0 < period_idx <= max_len:
            t = t[:period_idx+1]
        if len(t) > max_len:
            t = t[:max_len].rstrip()
            if not t.endswith('.'):  # cosmetic
                t += '...'
        return t

    def _generate_distractors(self, current_node: MCTSNode, tree, full_path: List[MCTSNode] = None) -> List[str]:
        distractors: List[str] = []
        try:
            nodes_by_distance = tree.find_same_level_nodes_by_distance(current_node, max_distance=10)
            for distance in sorted(nodes_by_distance.keys()):
                for node in nodes_by_distance[distance]:
                    if (node.node_id != current_node.node_id and node.answer != current_node.answer and node.answer not in distractors):
                        distractors.append(node.answer)
                    if len(distractors) >= 3:
                        break
                if len(distractors) >= 3:
                    break
        except Exception:
            if full_path:
                current_level = current_node.level
                for node in full_path:
                    if (node.level == current_level and node.node_id != current_node.node_id and node.answer != current_node.answer and node.answer not in distractors):
                        distractors.append(node.answer)
                        if len(distractors) >= 3:
                            break
        if len(distractors) < 3:
            question_lower = current_node.question.lower()
            if any(word in question_lower for word in ['color', 'what color']):
                color_distractors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown']
                for color in color_distractors:
                    if (color.lower() not in current_node.answer.lower() and color not in distractors):
                        distractors.append(color)
                        if len(distractors) >= 3:
                            break
            elif any(word in question_lower for word in ['how many', 'number', 'count']):
                try:
                    if current_node.answer.isdigit():
                        correct_num = int(current_node.answer)
                        num_distractors = [str(correct_num + i) for i in [-2, -1, 1, 2] if correct_num + i > 0]
                        for num_dist in num_distractors:
                            if num_dist not in distractors:
                                distractors.append(num_dist)
                except Exception:
                    pass
        return distractors

    def _generate_generic_distractor(self, current_node: MCTSNode, existing_options: List[Dict[str, Any]]) -> str:
        existing_texts = {opt['text'].lower() for opt in existing_options}
        generic_by_level = {
            1: [
                "Cannot be determined from the image",
                "Not visible in the image",
                "Multiple options present",
                "Unclear from this view",
            ],
            2: [
                "No clear relationship exists",
                "The connection is ambiguous",
                "Multiple interpretations possible",
                "Insufficient context provided",
            ],
            3: [
                "The analysis is inconclusive",
                "Multiple factors need consideration",
                "The reasoning is not supported",
                "Alternative explanations exist",
            ],
        }
        level_distractors = generic_by_level.get(current_node.level, generic_by_level[1])
        for distractor in level_distractors:
            if distractor.lower() not in existing_texts:
                return distractor
        return f"Option {len(existing_options) + 1}"

    def _generate_generic_filler(self, level: int = 1, existing_texts: Optional[set] = None) -> str:
        generic_by_level = {
            1: [
                "Not visible in the image",
                "Cannot be determined",
                "Ambiguous from this view",
                "Unclear detail",
            ],
            2: [
                "No clear relationship",
                "Multiple interpretations",
                "Insufficient context",
                "Connection is ambiguous",
            ],
            3: [
                "The conclusion is unsupported",
                "Alternative explanations exist",
                "Analysis is inconclusive",
                "Not strongly justified",
            ],
        }
        pool = generic_by_level.get(level, generic_by_level[1])
        existing_texts = existing_texts or set()
        for cand in pool:
            if cand.lower() not in existing_texts:
                return cand
        i = 1
        while True:
            cand = f"Alternative {i}"
            if cand.lower() not in existing_texts:
                return cand
            i += 1

    def save_sharegpt_data(self, conversations: List[Dict[str, Any]], output_path: str) -> None:
        import json
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved {len(conversations)} ShareGPT conversations to {output_path}")
        except Exception as e:
            logger.error(f"Error saving ShareGPT data: {e}")
            raise

    def get_conversion_statistics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not conversations:
            return {}
        total_conversations = len(conversations)
        total_turns = sum(len(conv['conversations']) for conv in conversations)
        total_questions = total_turns // 2
        conversation_lengths = [len(conv['conversations']) // 2 for conv in conversations]
        avg_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
        all_levels = []
        for conv in conversations:
            if 'metadata' in conv and 'levels' in conv['metadata']:
                all_levels.extend(conv['metadata']['levels'])
        level_distribution: Dict[int, int] = {}
        for level in all_levels:
            level_distribution[level] = level_distribution.get(level, 0) + 1
        return {
            "total_conversations": total_conversations,
            "total_questions": total_questions,
            "total_turns": total_turns,
            "average_questions_per_conversation": avg_length,
            "level_distribution": level_distribution,
            "max_conversation_length": max(conversation_lengths) if conversation_lengths else 0,
            "min_conversation_length": min(conversation_lengths) if conversation_lengths else 0,
        }

