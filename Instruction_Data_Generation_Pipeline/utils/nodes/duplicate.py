"""
Duplicate Q&A checker.
"""

import threading


class QADuplicateChecker:
    def __init__(self):
        self.qa_pairs = set()
        self.question_hashes = set()
        self.answer_hashes = set()
        self._lock = threading.RLock()

    def add_qa_pair(self, question: str, answer: str) -> None:
        qa_key = self._get_qa_key(question, answer)
        with self._lock:
            self.qa_pairs.add(qa_key)
            self.question_hashes.add(self._normalize_text(question))
            self.answer_hashes.add(self._normalize_text(answer))

    def is_duplicate(self, question: str, answer: str) -> bool:
        qa_key = self._get_qa_key(question, answer)
        with self._lock:
            return qa_key in self.qa_pairs

    def has_similar_question(self, question: str, threshold: float = 0.8, config: dict = None) -> bool:
        """Return True if any existing question is similar at given threshold.

        - Exact normalized match short-circuits to True.
        - If threshold >= 1.0, only exact match qualifies.
        - Otherwise, use difflib.SequenceMatcher ratio >= threshold.
        """
        # Get threshold from config if provided
        if config:
            try:
                duplicate_config = config.get("tree", {}).get("duplicate", {})
                threshold = float(duplicate_config.get("question_similarity_threshold", threshold))
            except Exception:
                pass
        normalized_question = self._normalize_text(question)
        with self._lock:
            if normalized_question in self.question_hashes:
                return True
            if threshold >= 1.0:
                return False
            try:
                from difflib import SequenceMatcher
                for q in self.question_hashes:
                    if not q or not normalized_question:
                        continue
                    if SequenceMatcher(None, normalized_question, q).ratio() >= threshold:
                        return True
            except Exception:
                return False
        return False

    def has_similar_answer(self, answer: str, threshold: float = 0.8, config: dict = None) -> bool:
        """Return True if any existing answer is similar at given threshold.

        Behavior mirrors has_similar_question.
        """
        # Get threshold from config if provided
        if config:
            try:
                duplicate_config = config.get("tree", {}).get("duplicate", {})
                threshold = float(duplicate_config.get("answer_similarity_threshold", threshold))
            except Exception:
                pass
        normalized_answer = self._normalize_text(answer)
        with self._lock:
            if normalized_answer in self.answer_hashes:
                return True
            if threshold >= 1.0:
                return False
            try:
                from difflib import SequenceMatcher
                for a in self.answer_hashes:
                    if not a or not normalized_answer:
                        continue
                    if SequenceMatcher(None, normalized_answer, a).ratio() >= threshold:
                        return True
            except Exception:
                return False
        return False

    def _get_qa_key(self, question: str, answer: str) -> str:
        return f"{self._normalize_text(question)}|{self._normalize_text(answer)}"

    def _normalize_text(self, text: str) -> str:
        import re
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        normalized = " ".join(normalized.split())
        return normalized

    def clear(self) -> None:
        with self._lock:
            self.qa_pairs.clear()
            self.question_hashes.clear()
            self.answer_hashes.clear()
