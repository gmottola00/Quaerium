"""LLM-based metadata extraction from text.

This module provides a generic metadata extractor that uses LLMs to extract
structured information from unstructured text using customizable prompts.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from quaerium.core.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LLMMetadataExtractor:
    """Extract structured metadata from text using LLM prompts.

    Generic implementation that accepts custom prompt templates for different domains.
    Useful for extracting entities, dates, classifications, or any structured data
    from unstructured documents.

    Examples:
        Legal Domain:
            >>> from quaerium.infra.llm import OllamaLLMClient
            >>>
            >>> LEGAL_SYSTEM_PROMPT = '''
            ... You are a legal document analyzer. Extract metadata in JSON format:
            ... {"case_number": "", "court": "", "date": "", "parties": []}
            ... '''
            >>>
            >>> LEGAL_EXTRACTION_PROMPT = '''
            ... Given this legal document text:
            ... {context}
            ...
            ... Extract: case number, court name, filing date, and party names.
            ... Return only valid JSON.
            ... '''
            >>>
            >>> llm = OllamaLLMClient(model="llama3.2")
            >>> extractor = LLMMetadataExtractor(
            ...     llm_client=llm,
            ...     system_prompt=LEGAL_SYSTEM_PROMPT,
            ...     extraction_prompt_template=LEGAL_EXTRACTION_PROMPT,
            ... )
            >>> metadata = extractor.extract(document_text)
            >>> print(metadata["case_number"])

        Tender Domain:
            >>> TENDER_SYSTEM_PROMPT = '''
            ... You are an assistant for analyzing tender documents.
            ... Extract metadata in JSON: {"ente_appaltante": "", "cig": "", "importo": ""}
            ... '''
            >>>
            >>> extractor = LLMMetadataExtractor(llm, TENDER_SYSTEM_PROMPT, ...)
            >>> metadata = extractor.extract(tender_text)

    Attributes:
        llm_client: LLM client implementing LLMClient protocol
        system_prompt: System prompt defining extraction task and output format
        extraction_prompt_template: User prompt template with {context} placeholder
        max_text_length: Maximum text length to send to LLM (truncates longer texts)
        response_parser: Optional custom parser for LLM response (default: JSON parser)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        extraction_prompt_template: str,
        *,
        max_text_length: int = 8000,
        response_parser: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize metadata extractor.

        Args:
            llm_client: LLM client for generation
            system_prompt: System prompt defining extraction schema
            extraction_prompt_template: Prompt template with {context} placeholder
            max_text_length: Max characters to send to LLM (default: 8000)
            response_parser: Custom parser function (default: JSON parser with cleanup)

        Raises:
            ValueError: If extraction_prompt_template doesn't contain {context}
        """
        if "{context}" not in extraction_prompt_template:
            raise ValueError(
                "extraction_prompt_template must contain {context} placeholder"
            )

        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.extraction_prompt_template = extraction_prompt_template
        self.max_text_length = max_text_length
        self.response_parser = response_parser or self._default_json_parser

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text using LLM.

        Args:
            text: Input text to extract metadata from

        Returns:
            Dictionary of extracted metadata. Returns empty dict if parsing fails.

        Example:
            >>> metadata = extractor.extract("Contract between Acme Corp and...")
            >>> print(metadata["case_number"])
            "2024-CV-12345"
        """
        # Truncate text to max length
        truncated_text = text[: self.max_text_length]

        # Format prompt with text
        user_prompt = self.extraction_prompt_template.format(
            context=truncated_text
        )

        # Generate with LLM
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.0,  # Use deterministic generation for extraction
            )
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            return {}

        # Parse response
        return self.response_parser(response)

    @staticmethod
    def _default_json_parser(raw_response: str) -> Dict[str, Any]:
        """Default parser that cleans and parses JSON from LLM response.

        Handles common LLM output formats:
        - Wrapped in ```json``` code blocks
        - Wrapped in ``` code blocks
        - Raw JSON

        Args:
            raw_response: Raw LLM response string

        Returns:
            Parsed dictionary or empty dict if parsing fails
        """
        # Clean code block markers
        cleaned = raw_response.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

        # Parse JSON
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Failed to parse JSON from LLM response: {exc}\n"
                f"Response: {raw_response[:200]}..."
            )
            return {}


__all__ = ["LLMMetadataExtractor"]
