"""Tests for LLMMetadataExtractor."""

import pytest
from unittest.mock import Mock

from quaerium.core.metadata import LLMMetadataExtractor


class TestLLMMetadataExtractor:
    """Test LLM-based metadata extraction."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        mock = Mock()
        mock.generate.return_value = '{"key": "value"}'
        return mock

    @pytest.fixture
    def extractor(self, mock_llm_client):
        """Create extractor with mock LLM."""
        return LLMMetadataExtractor(
            llm_client=mock_llm_client,
            system_prompt="Extract metadata",
            extraction_prompt_template="From text: {context}\nExtract metadata.",
        )

    def test_extract_valid_json(self, extractor, mock_llm_client):
        """Test extraction with valid JSON response."""
        mock_llm_client.generate.return_value = '{"author": "John Doe", "year": "2024"}'

        metadata = extractor.extract("Some document text")

        assert metadata == {"author": "John Doe", "year": "2024"}
        mock_llm_client.generate.assert_called_once()

    def test_extract_json_with_code_blocks(self, extractor, mock_llm_client):
        """Test extraction with JSON in code blocks."""
        mock_llm_client.generate.return_value = '```json\n{"key": "value"}\n```'

        metadata = extractor.extract("Text")

        assert metadata == {"key": "value"}

    def test_extract_json_with_plain_code_blocks(self, extractor, mock_llm_client):
        """Test extraction with JSON in plain code blocks."""
        mock_llm_client.generate.return_value = '```\n{"key": "value"}\n```'

        metadata = extractor.extract("Text")

        assert metadata == {"key": "value"}

    def test_extract_malformed_json(self, extractor, mock_llm_client):
        """Test extraction with malformed JSON returns empty dict."""
        mock_llm_client.generate.return_value = 'Not valid JSON {{'

        metadata = extractor.extract("Text")

        assert metadata == {}

    def test_extract_truncates_long_text(self, extractor, mock_llm_client):
        """Test that long text is truncated."""
        long_text = "x" * 10000
        extractor.max_text_length = 100

        extractor.extract(long_text)

        # Check prompt contains truncated text
        call_args = mock_llm_client.generate.call_args
        # The prompt should contain "From text: " + 100 chars + "\nExtract metadata."
        assert "x" * 100 in call_args.kwargs["prompt"]
        assert "x" * 101 not in call_args.kwargs["prompt"]

    def test_extract_uses_temperature_zero(self, extractor, mock_llm_client):
        """Test that extraction uses temperature=0."""
        extractor.extract("Text")

        call_args = mock_llm_client.generate.call_args
        assert call_args.kwargs.get("temperature") == 0.0

    def test_extract_uses_system_prompt(self, extractor, mock_llm_client):
        """Test that system prompt is passed to LLM."""
        extractor.extract("Text")

        call_args = mock_llm_client.generate.call_args
        assert call_args.kwargs.get("system_prompt") == "Extract metadata"

    def test_custom_response_parser(self, mock_llm_client):
        """Test custom response parser."""
        def custom_parser(response: str) -> dict:
            # Parse "KEY=value" format
            result = {}
            for line in response.split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    result[k.strip()] = v.strip()
            return result

        extractor = LLMMetadataExtractor(
            llm_client=mock_llm_client,
            system_prompt="Extract",
            extraction_prompt_template="{context}",
            response_parser=custom_parser,
        )

        mock_llm_client.generate.return_value = "author=John\nyear=2024"
        metadata = extractor.extract("Text")

        assert metadata == {"author": "John", "year": "2024"}

    def test_missing_context_placeholder_raises_error(self, mock_llm_client):
        """Test that missing {context} raises ValueError."""
        with pytest.raises(ValueError, match="must contain {context}"):
            LLMMetadataExtractor(
                llm_client=mock_llm_client,
                system_prompt="Extract",
                extraction_prompt_template="No placeholder here",
            )

    def test_llm_exception_returns_empty_dict(self, extractor, mock_llm_client):
        """Test that LLM exceptions are handled gracefully."""
        mock_llm_client.generate.side_effect = RuntimeError("LLM error")

        metadata = extractor.extract("Text")

        assert metadata == {}

    def test_empty_json_response(self, extractor, mock_llm_client):
        """Test extraction with empty JSON object."""
        mock_llm_client.generate.return_value = '{}'

        metadata = extractor.extract("Text")

        assert metadata == {}

    def test_nested_json_response(self, extractor, mock_llm_client):
        """Test extraction with nested JSON."""
        mock_llm_client.generate.return_value = '{"metadata": {"author": "John", "year": 2024}}'

        metadata = extractor.extract("Text")

        assert metadata == {"metadata": {"author": "John", "year": 2024}}

    def test_extract_formats_prompt_correctly(self, extractor, mock_llm_client):
        """Test that prompt is formatted with context."""
        test_text = "This is a test document"

        extractor.extract(test_text)

        call_args = mock_llm_client.generate.call_args
        expected_prompt = f"From text: {test_text}\nExtract metadata."
        assert call_args.kwargs["prompt"] == expected_prompt
