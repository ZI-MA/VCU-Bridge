import re


def clean_model_response(content: str) -> str:
    """Clean model response."""
    content = re.sub(r'<THINK>.*?</THINK>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content = re.sub(r'`([^`]+)`', r'\1', content)
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    content = re.sub(r'__([^_]+)__', r'\1', content)
    content = re.sub(r'\*([^*]+)\*', r'\1', content)
    content = re.sub(r'_([^_]+)_', r'\1', content)
    content = re.sub(r'~~([^~]+)~~', r'\1', content)
    content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
    return content.strip().upper()


def extract_answer_simple(text: str) -> str | None:
    """Direct match."""
    if text in ['A', 'B', 'C', 'D']:
        return text
    return None


def extract_answer_with_dot(text: str) -> str | None:
    """Match with dot (e.g., A.)"""
    cleaned = re.sub(r'[^A-D]', '', text)
    if not cleaned:
        return None
    first_char = cleaned[0]
    char_position = text.find(first_char)
    if char_position + 1 < len(text) and text[char_position + 1] == '.':
        return first_char
    return None


def has_negation_near_match(text: str, match_start: int) -> bool:
    """Check for negation words near match."""
    NEGATION_PATTERNS = [
        r'(?:NOT|WRONG|INCORRECT|ISN\'T|AREN\'T)\s+(?:THE\s+)?(?:ANSWER|OPTION|CHOICE)',
        r'DON\'T',
        r'(?:ANSWER|OPTION)\s*[:：]?\s*[A-D]\s+IS\s+(?:WRONG|INCORRECT)',
    ]
    check_start = max(0, match_start - 50)
    check_end = min(len(text), match_start + 20)
    context = text[check_start:check_end]
    
    for pattern in NEGATION_PATTERNS:
        for negation_match in re.finditer(pattern, context):
            negation_end_absolute = check_start + negation_match.end()
            negation_start_absolute = check_start + negation_match.start()
            if negation_start_absolute >= match_start - 20 and negation_end_absolute <= match_start + 15:
                distance = match_start - negation_end_absolute
                if -10 <= distance <= 15:
                    return True
    return False


def extract_answer_with_pattern(text: str) -> str | None:
    """Pattern matching with negation filter."""
    ANSWER_PATTERNS = [
        r'ANSWER\s*[:：]\s*([A-D])(?:\s*[.,，。]|\s*$|\n)',
        r'ANSWER\s+IS:?\s*([A-D])(?:\s*[.,，。]|\s*$|\n)',
        r'(?:OPTION|CHOICE)\s+IS:?\s*([A-D])(?:\s*[.,，。]|\s*$|\n)',
        r'(?:CHOOSE|SELECT)(?:\s+OPTION)?\s+([A-D])(?:\s*[.,，。]|\s*$|\n)',
    ]
    all_matches = []
    for pattern in ANSWER_PATTERNS:
        all_matches.extend(re.finditer(pattern, text))
    if not all_matches:
        return None
    all_matches.sort(key=lambda x: x.start(), reverse=True)
    for match in all_matches:
        if not has_negation_near_match(text, match.start()):
            return match.group(1)
    return None


def extract_answer_from_last_line(text: str) -> str | None:
    """Extract from last line."""
    lines = text.strip().split('\n')
    if not lines:
        return None
    last_line = lines[-1].strip()
    if last_line in ['A', 'B', 'C', 'D'] and len(lines) >= 2:
        second_last = lines[-2].strip()
        if second_last not in ['A', 'B', 'C', 'D']:
            return last_line
    match = re.match(r'^([A-D])\.', last_line)
    if match:
        if len(lines) >= 2:
            second_last = lines[-2].strip()
            if not re.match(r'^([A-D])\.', second_last):
                return match.group(1)
        else:
            return match.group(1)
    return None


def parse_multiple_choice_answer(content: str) -> str:
    """Parse multiple-choice answer from model output. Returns A/B/C/D."""
    if not content:
        raise ValueError("Empty content")
    
    original_content = content
    cleaned = clean_model_response(content)
    
    if not re.search(r'[A-D]', cleaned):
        raise ValueError(f"Invalid response format: '{original_content}'")
    
    result = (
        extract_answer_simple(cleaned) or
        extract_answer_with_dot(cleaned) or
        extract_answer_with_pattern(cleaned) or
        extract_answer_from_last_line(cleaned)
    )
    
    if result:
        return result
    
    raise ValueError(f"Invalid response format: '{original_content}'")
