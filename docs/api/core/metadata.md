# Metadata API Reference

Complete API documentation for metadata extraction and enrichment components.

## LLMMetadataExtractor

::: quaerium.core.metadata.LLMMetadataExtractor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members_order: source
      group_by_category: true

## MetadataEnricher

::: quaerium.core.chunking.MetadataEnricher
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members_order: source
      group_by_category: true

## Usage Examples

### LLMMetadataExtractor

```python
from quaerium.core.metadata import LLMMetadataExtractor
from quaerium.infra.llm import OllamaLLMClient

# Initialize
llm = OllamaLLMClient(model="phi3:mini")
extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt="Extract metadata in JSON...",
    extraction_prompt_template="From text:\n{context}\n\nExtract metadata.",
    max_text_length=8000,
)

# Extract
metadata = extractor.extract("Document text...")
# Returns: Dict[str, Any]
```

### MetadataEnricher

```python
from quaerium.core.chunking import MetadataEnricher

# Initialize
enricher = MetadataEnricher(
    format_template="[{key}: {value}]",
    excluded_keys=["id", "chunk_id"],
    separator=" ",
)

# Enrich single text
text = "Original chunk text"
metadata = {"author": "John", "year": "2024"}
enriched = enricher.enrich_text(text, metadata)
# Returns: "Original chunk text [author: John] [year: 2024]"

# Enrich batch of chunks
enriched_texts = enricher.enrich_chunks(chunks)
# Returns: List[str]
```

## Type Definitions

### LLMMetadataExtractor Types

```python
from typing import Dict, Any, Callable

# Constructor parameters
llm_client: LLMClient
system_prompt: str
extraction_prompt_template: str
max_text_length: int = 8000
response_parser: Optional[Callable[[str], Dict[str, Any]]] = None

# Return types
extract(text: str) -> Dict[str, Any]
```

### MetadataEnricher Types

```python
from typing import List, Set, Dict, Any

# Constructor parameters
excluded_keys: List[str] | None = None
format_template: str = "[{key}: {value}]"
separator: str = " "

# Return types
enrich_text(text: str, metadata: Dict[str, Any]) -> str
enrich_chunks(chunks: List[TokenChunkLike]) -> List[str]
```

## Error Handling

### LLMMetadataExtractor

```python
# Returns empty dict on error
try:
    metadata = extractor.extract(text)
except Exception:
    # LLM failures are caught internally
    pass

if not metadata:
    # Handle extraction failure
    print("Extraction failed or returned no results")
```

### MetadataEnricher

```python
# Raises ValueError on invalid configuration
try:
    enricher = MetadataEnricher(
        format_template="[{key}]"  # Missing {value}
    )
except ValueError as e:
    print(f"Invalid template: {e}")
```

## Performance Considerations

### LLMMetadataExtractor

- **Text truncation**: Long documents are truncated to `max_text_length`
- **Temperature**: Uses `temperature=0.0` for deterministic extraction
- **Parsing**: JSON parsing with code block cleanup is automatic

### MetadataEnricher

- **Batch processing**: Use `enrich_chunks()` for efficiency
- **Filtering**: Default excluded keys: `file_name`, `chunk_id`, `id`, `source_chunk_id`
- **String values only**: Non-string metadata values are ignored

## See Also

- [User Guide](../../guides/metadata_extraction.md): Conceptual overview and best practices
- [Examples](../../examples/metadata_extraction.md): Complete working examples
- [Protocols](./protocols.md): Core protocol definitions
- [Types](./types.md): Core type definitions
