"""
Tests for dataset loading and management.

Tests JSONL dataset loader and validation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from quaerium.core.evaluation.dataset import EvaluationDataset
from quaerium.infra.evaluation.datasets import JSONLDataset


class TestJSONLDataset:
    """Test JSONL dataset loader."""

    def test_load_valid_dataset(self, tmp_path):
        """Test loading valid JSONL dataset."""
        # Create temporary JSONL file
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test1", "relevant_doc_ids": ["doc1", "doc2"]}\n')
            f.write(
                '{"query": "test2", "relevant_doc_ids": ["doc3"], "reference_answer": "answer2"}\n'
            )

        dataset = JSONLDataset(dataset_file)

        assert len(dataset) == 2
        assert dataset[0].query == "test1"
        assert dataset[0].relevant_doc_ids == ["doc1", "doc2"]
        assert dataset[0].reference_answer is None

        assert dataset[1].query == "test2"
        assert dataset[1].relevant_doc_ids == ["doc3"]
        assert dataset[1].reference_answer == "answer2"

    def test_load_with_metadata(self, tmp_path):
        """Test loading examples with metadata."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write(
                '{"query": "test", "relevant_doc_ids": ["doc1"], "metadata": {"category": "tech", "difficulty": "easy"}}\n'
            )

        dataset = JSONLDataset(dataset_file)

        assert dataset[0].metadata == {"category": "tech", "difficulty": "easy"}

    def test_iterate_dataset(self, tmp_path):
        """Test iterating over dataset."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test1", "relevant_doc_ids": ["doc1"]}\n')
            f.write('{"query": "test2", "relevant_doc_ids": ["doc2"]}\n')

        dataset = JSONLDataset(dataset_file)

        queries = [example.query for example in dataset]
        assert queries == ["test1", "test2"]

    def test_dataset_metadata(self, tmp_path):
        """Test dataset metadata generation."""
        dataset_file = tmp_path / "my_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test", "relevant_doc_ids": ["doc1"]}\n')

        dataset = JSONLDataset(dataset_file)

        metadata = dataset.metadata
        assert metadata.name == "my_dataset"
        assert metadata.num_examples == 1
        assert "evaluation" in metadata.tags
        assert "jsonl" in metadata.tags

    def test_skip_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test1", "relevant_doc_ids": ["doc1"]}\n')
            f.write("\n")  # Empty line
            f.write('{"query": "test2", "relevant_doc_ids": ["doc2"]}\n')

        dataset = JSONLDataset(dataset_file)

        assert len(dataset) == 2

    def test_skip_invalid_json(self, tmp_path):
        """Test that invalid JSON lines are skipped."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test1", "relevant_doc_ids": ["doc1"]}\n')
            f.write("invalid json line\n")
            f.write('{"query": "test2", "relevant_doc_ids": ["doc2"]}\n')

        dataset = JSONLDataset(dataset_file)

        assert len(dataset) == 2

    def test_skip_missing_query(self, tmp_path):
        """Test that examples without query are skipped."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"relevant_doc_ids": ["doc1"]}\n')  # Missing query
            f.write('{"query": "test2", "relevant_doc_ids": ["doc2"]}\n')

        dataset = JSONLDataset(dataset_file)

        assert len(dataset) == 1
        assert dataset[0].query == "test2"

    def test_skip_empty_relevant_docs(self, tmp_path):
        """Test that examples without relevant docs are skipped."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test1", "relevant_doc_ids": []}\n')  # Empty list
            f.write('{"query": "test2", "relevant_doc_ids": ["doc2"]}\n')

        dataset = JSONLDataset(dataset_file)

        assert len(dataset) == 1
        assert dataset[0].query == "test2"

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            JSONLDataset("nonexistent_file.jsonl")

    def test_empty_file(self, tmp_path):
        """Test that ValueError is raised for empty file."""
        dataset_file = tmp_path / "empty.jsonl"
        dataset_file.touch()

        with pytest.raises(ValueError, match="empty"):
            JSONLDataset(dataset_file)

    def test_index_access(self, tmp_path):
        """Test accessing examples by index."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test1", "relevant_doc_ids": ["doc1"]}\n')
            f.write('{"query": "test2", "relevant_doc_ids": ["doc2"]}\n')
            f.write('{"query": "test3", "relevant_doc_ids": ["doc3"]}\n')

        dataset = JSONLDataset(dataset_file)

        assert dataset[0].query == "test1"
        assert dataset[1].query == "test2"
        assert dataset[2].query == "test3"
        assert dataset[-1].query == "test3"

    def test_protocol_compliance(self, tmp_path):
        """Test that JSONLDataset satisfies protocol."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"query": "test", "relevant_doc_ids": ["doc1"]}\n')

        dataset = JSONLDataset(dataset_file)
        assert isinstance(dataset, EvaluationDataset)

    def test_load_sample_dataset(self):
        """Test loading the sample dataset (if it exists)."""
        sample_path = Path("examples/data/sample_eval_dataset.jsonl")

        if sample_path.exists():
            dataset = JSONLDataset(sample_path)

            assert len(dataset) > 0
            assert dataset.metadata.name == "sample_eval_dataset"

            # Verify first example
            first_example = dataset[0]
            assert first_example.query
            assert first_example.relevant_doc_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
