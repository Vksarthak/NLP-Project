"""
DocumentAnalyzer - Orchestrates NLP analysis tasks using LLM.

This module coordinates document analysis by combining the LLM client,
prompt manager, and document processor to perform summarization,
entity extraction, sentiment analysis, topic modeling, and Q&A.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.llm_client import GeminiClient
from src.prompt_manager import PromptManager
from src.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Orchestrator for NLP analysis using LLM.

    Coordinates multiple NLP tasks by rendering prompt templates,
    calling the Gemini API, and parsing structured JSON responses.
    Handles long documents by automatic chunking.
    """

    MAX_CONTEXT_CHARS = 15000  # Character limit for a single API call

    def __init__(
        self,
        llm_client: GeminiClient,
        prompt_manager: PromptManager,
    ) -> None:
        """Initialize the DocumentAnalyzer.

        Args:
            llm_client: An initialized GeminiClient instance.
            prompt_manager: An initialized PromptManager instance.
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.doc_processor = DocumentProcessor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from an LLM response.

        Handles common LLM output quirks such as markdown code fences
        and leading/trailing prose around the JSON object.

        Args:
            response: Raw text returned by the LLM.

        Returns:
            Parsed dictionary. On failure a fallback dict with the raw text
            is returned so callers never receive ``None``.
        """
        # Try to find a JSON block inside markdown code fences first.
        json_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE
        )
        json_str = json_match.group(1) if json_match else response

        try:
            json_str = json_str.strip()
            # Locate the outermost curly braces.
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                json_str = json_str[start : end + 1]
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON from LLM response: %s", exc)
            return {"error": "Failed to parse structured response", "raw_response": response}

    def _prepare_text(self, text: str) -> str:
        """Truncate text to fit the context window if necessary.

        If the text exceeds ``MAX_CONTEXT_CHARS`` the first chunk produced
        by :class:`DocumentProcessor` is returned with a truncation note.

        Args:
            text: Full document text.

        Returns:
            Text suitable for inclusion in a single prompt.
        """
        if len(text) > self.MAX_CONTEXT_CHARS:
            logger.info(
                "Document text (%d chars) exceeds limit (%d). Truncating.",
                len(text),
                self.MAX_CONTEXT_CHARS,
            )
            chunks = self.doc_processor.chunk_text(
                text, chunk_size=self.MAX_CONTEXT_CHARS
            )
            if chunks:
                return chunks[0] + "\n\n[Note: Document was truncated for analysis. Only the first section was used.]"
        return text

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def summarize(self, text: str) -> Dict[str, Any]:
        """Generate a structured summary of the document.

        Args:
            text: Document text.

        Returns:
            Parsed JSON dict with ``summary`` and ``key_points``.
        """
        prepared = self._prepare_text(text)
        prompt = self.prompt_manager.render("summarize", document_text=prepared)
        response = self.llm_client.generate(prompt)
        return self._parse_json_response(response)

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from the document.

        Args:
            text: Document text.

        Returns:
            Parsed JSON dict with entity categories.
        """
        prepared = self._prepare_text(text)
        prompt = self.prompt_manager.render("extract_entities", document_text=prepared)
        response = self.llm_client.generate(prompt)
        return self._parse_json_response(response)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and tone of the document.

        Args:
            text: Document text.

        Returns:
            Parsed JSON dict with ``sentiment``, ``tones``, ``explanation``.
        """
        prepared = self._prepare_text(text)
        prompt = self.prompt_manager.render("sentiment", document_text=prepared)
        response = self.llm_client.generate(prompt)
        return self._parse_json_response(response)

    def extract_topics(self, text: str) -> Dict[str, Any]:
        """Extract key topics and themes from the document.

        Args:
            text: Document text.

        Returns:
            Parsed JSON dict with ``topics`` list.
        """
        prepared = self._prepare_text(text)
        prompt = self.prompt_manager.render("topics", document_text=prepared)
        response = self.llm_client.generate(prompt)
        return self._parse_json_response(response)

    def ask_question(
        self,
        text: str,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Answer a question about the document.

        Args:
            text: Document text used as context.
            question: The user's question.
            chat_history: Optional previous conversation turns.

        Returns:
            Parsed JSON dict with ``chain_of_thought`` and ``answer``.
        """
        prepared = self._prepare_text(text)

        # Build a readable history string if available.
        history_str = ""
        if chat_history:
            history_str = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in chat_history
            )

        prompt = self.prompt_manager.render(
            "qa",
            document_text=prepared,
            user_question=question,
        )
        response = self.llm_client.generate(prompt)
        return self._parse_json_response(response)

    def full_analysis(self, text: str) -> Dict[str, Any]:
        """Run all analysis tasks and return combined results.

        Executes summarization, entity extraction, sentiment analysis,
        and topic extraction sequentially.

        Args:
            text: Document text.

        Returns:
            Dictionary keyed by analysis type with parsed results.
        """
        logger.info("Starting full document analysis")
        results: Dict[str, Any] = {
            "summary": self.summarize(text),
            "entities": self.extract_entities(text),
            "sentiment": self.analyze_sentiment(text),
            "topics": self.extract_topics(text),
        }
        logger.info("Full analysis complete (%d API calls total)", self.llm_client.total_calls)
        return results
