# Metadata Extraction Examples

This page provides complete, runnable examples for metadata extraction and enrichment across different domains.

## Prerequisites

```bash
# Install quaerium
pip install quaerium

# Start Ollama (if using local LLMs)
ollama serve

# Pull required models
ollama pull phi3:mini
ollama pull nomic-embed-text
```

## Example 1: Legal Documents

Extract case information from legal filings and enrich chunks for retrieval.

```python
from quaerium.core.metadata import LLMMetadataExtractor
from quaerium.core.chunking import MetadataEnricher, TokenChunker
from quaerium.infra.llm import OllamaLLMClient
from quaerium.infra.embedding import OllamaEmbeddingClient

# ============================================================================
# Define Legal Domain Prompts
# ============================================================================

LEGAL_SYSTEM_PROMPT = """
You are a legal document analyzer. Extract metadata in this JSON format:
{
  "case_number": "",
  "court": "",
  "filing_date": "",
  "parties": {"plaintiff": "", "defendant": ""},
  "document_type": ""
}

Rules:
- Return ONLY valid JSON
- Use empty strings if field not found
- Be precise with dates (YYYY-MM-DD format)
"""

LEGAL_EXTRACTION_PROMPT = """
Analyze this legal document:

{context}

Extract: case number, court name, filing date, parties, and document type.
Return JSON only.
"""

# ============================================================================
# Sample Legal Document
# ============================================================================

SAMPLE_CONTRACT = """
SUPERIOR COURT OF CALIFORNIA
COUNTY OF LOS ANGELES

Case No. 2024-CV-12345

John Doe, Plaintiff
v.
Acme Corporation, Defendant

Filed: January 15, 2024

COMPLAINT FOR BREACH OF CONTRACT

Plaintiff alleges that Defendant breached the employment agreement
entered into on March 1, 2023. The contract stipulated a two-year term
with specific performance milestones. Defendant terminated Plaintiff's
employment without cause on November 30, 2023.

Plaintiff seeks damages of $250,000 for lost wages and benefits.
"""

# ============================================================================
# Extract and Enrich
# ============================================================================

# Initialize components
llm = OllamaLLMClient(model="phi3:mini")
embed_client = OllamaEmbeddingClient(model="nomic-embed-text")

# Create extractor
extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=LEGAL_SYSTEM_PROMPT,
    extraction_prompt_template=LEGAL_EXTRACTION_PROMPT,
)

# Extract metadata
metadata = extractor.extract(SAMPLE_CONTRACT)
print(f"Extracted metadata: {metadata}")
# Output: {'case_number': '2024-CV-12345', 'court': 'Superior Court...', ...}

# Create chunks (simplified - in production use DynamicChunker)
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    section_path: str = ""
    page_numbers: List[int] = field(default_factory=list)
    source_chunk_id: str = ""
    
    def to_dict(self):
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

paragraphs = [p.strip() for p in SAMPLE_CONTRACT.split('\n\n') if p.strip()]
chunks = [
    Chunk(id=f"chunk-{i}", text=para, metadata=metadata.copy())
    for i, para in enumerate(paragraphs)
]

# Enrich chunks
enricher = MetadataEnricher()
enriched_texts = enricher.enrich_chunks(chunks)

print(f"\nEnriched example:")
print(f"Original: {chunks[3].text[:60]}...")
print(f"Enriched: {enriched_texts[3][:120]}...")

# Generate embeddings
embeddings = embed_client.embed_batch(enriched_texts)
print(f"\nGenerated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")

# Now you can index these in your vector store
# vector_db.index(chunks, embeddings)
```

**Output:**
```
Extracted metadata: {
  'case_number': '2024-CV-12345',
  'court': 'Superior Court of California, County of Los Angeles',
  'filing_date': '2024-01-15',
  'parties': {'plaintiff': 'John Doe', 'defendant': 'Acme Corporation'},
  'document_type': 'complaint for breach of contract'
}

Enriched example:
Original: Plaintiff alleges that Defendant breached the employment...
Enriched: Plaintiff alleges... [case_number: 2024-CV-12345] [court: Superior Court...]...

Generated 8 embeddings (dim: 768)
```

## Example 2: Medical Records

Extract patient and visit information from medical notes.

```python
MEDICAL_SYSTEM_PROMPT = """
You are a medical records analyzer. Extract in JSON:
{
  "patient_id": "",
  "visit_date": "",
  "chief_complaint": "",
  "diagnosis_codes": [],
  "physician": ""
}

Use empty string/array if not found.
"""

MEDICAL_EXTRACTION_PROMPT = """
Medical record:
{context}

Extract patient ID, visit date, chief complaint, diagnosis codes, and physician.
Return JSON only.
"""

SAMPLE_RECORD = """
PATIENT ID: P-789456
DATE OF VISIT: 2024-02-10
PHYSICIAN: Dr. Sarah Johnson

CHIEF COMPLAINT: Patient reports persistent headaches for 3 weeks.

DIAGNOSIS:
- ICD-10 R51.9 - Headache, unspecified
- ICD-10 Z13.1 - Encounter for screening examination

TREATMENT PLAN: Prescribed medication, follow-up in 2 weeks.
"""

# Use same workflow as legal example
extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=MEDICAL_SYSTEM_PROMPT,
    extraction_prompt_template=MEDICAL_EXTRACTION_PROMPT,
)

metadata = extractor.extract(SAMPLE_RECORD)
print(metadata)
# {'patient_id': 'P-789456', 'visit_date': '2024-02-10', ...}
```

## Example 3: E-commerce Product Catalogs

Extract product specifications from descriptions.

```python
PRODUCT_SYSTEM_PROMPT = """
Extract product metadata in JSON:
{
  "sku": "",
  "category": "",
  "brand": "",
  "price": "",
  "specifications": {}
}
"""

PRODUCT_EXTRACTION_PROMPT = """
Product description:
{context}

Extract SKU, category, brand, price, and key specifications.
Return JSON only.
"""

SAMPLE_PRODUCT = """
SKU: LAPTOP-XPS-15-2024
Category: Electronics > Laptops
Brand: Dell

Dell XPS 15 (2024 Model)
Price: $1,899.99

Specifications:
- Display: 15.6" 4K OLED Touch
- Processor: Intel Core i7-13700H
- RAM: 32GB DDR5
- Storage: 1TB NVMe SSD
- Graphics: NVIDIA RTX 4060
- Battery: 86Wh, up to 12 hours
"""

extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=PRODUCT_SYSTEM_PROMPT,
    extraction_prompt_template=PRODUCT_EXTRACTION_PROMPT,
)

metadata = extractor.extract(SAMPLE_PRODUCT)
print(metadata)
# {'sku': 'LAPTOP-XPS-15-2024', 'brand': 'Dell', 'price': '$1,899.99', ...}
```

## Example 4: Academic Papers

Extract publication metadata from research papers.

```python
ACADEMIC_SYSTEM_PROMPT = """
Extract paper metadata in JSON:
{
  "title": "",
  "authors": [],
  "institution": "",
  "publication_date": "",
  "doi": "",
  "keywords": []
}
"""

ACADEMIC_EXTRACTION_PROMPT = """
Research paper:
{context}

Extract title, authors, institution, publication date, DOI, and keywords.
Return JSON only.
"""

SAMPLE_PAPER = """
Deep Learning for Natural Language Processing: A Survey

Authors: Alice Smith¹, Bob Johnson², Carol Williams¹
¹ MIT Computer Science and AI Lab
² Stanford NLP Group

Publication Date: January 2024
DOI: 10.1234/example.2024.001

Keywords: natural language processing, deep learning, transformers,
attention mechanism, large language models

ABSTRACT
This paper surveys recent advances in deep learning for NLP...
"""

extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=ACADEMIC_SYSTEM_PROMPT,
    extraction_prompt_template=ACADEMIC_EXTRACTION_PROMPT,
)

metadata = extractor.extract(SAMPLE_PAPER)
print(metadata)
# {'title': 'Deep Learning for NLP...', 'authors': ['Alice Smith', ...], ...}
```

## Example 5: Full Production Pipeline

Complete example with PDF parsing, chunking, extraction, enrichment, and indexing.

```python
from quaerium.core.chunking import DynamicChunker, TokenChunker, MetadataEnricher
from quaerium.core.metadata import LLMMetadataExtractor
from quaerium.infra.parsers import create_ingestion_service
from quaerium.infra.llm import OllamaLLMClient
from quaerium.infra.embedding import OllamaEmbeddingClient

def process_document(
    file_path: str,
    extractor: LLMMetadataExtractor,
    enricher: MetadataEnricher,
    embed_client: OllamaEmbeddingClient,
):
    """Process a document through the full pipeline."""
    
    # 1. Parse PDF/DOCX
    ingestion = create_ingestion_service()
    parsed_pages = ingestion.parse_file(file_path)
    full_text = " ".join([page["text"] for page in parsed_pages])
    
    # 2. Extract metadata from full document
    metadata = extractor.extract(full_text)
    print(f"✓ Extracted metadata: {list(metadata.keys())}")
    
    # 3. Chunk document
    dynamic_chunker = DynamicChunker()
    token_chunker = TokenChunker(max_tokens=512, overlap_tokens=50)
    
    structured_chunks = dynamic_chunker.build_chunks(parsed_pages)
    token_chunks = token_chunker.chunk(structured_chunks)
    print(f"✓ Created {len(token_chunks)} chunks")
    
    # 4. Add metadata to all chunks
    for chunk in token_chunks:
        chunk.metadata.update(metadata)
    
    # 5. Enrich chunks
    enriched_texts = enricher.enrich_chunks(token_chunks)
    print(f"✓ Enriched {len(enriched_texts)} chunks")
    
    # 6. Generate embeddings
    embeddings = embed_client.embed_batch(enriched_texts)
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    return token_chunks, embeddings

# Usage
llm = OllamaLLMClient(model="phi3:mini")
embed_client = OllamaEmbeddingClient(model="nomic-embed-text")

extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=LEGAL_SYSTEM_PROMPT,
    extraction_prompt_template=LEGAL_EXTRACTION_PROMPT,
)

enricher = MetadataEnricher()

chunks, embeddings = process_document(
    "contract.pdf",
    extractor,
    enricher,
    embed_client,
)

# Now index in vector store
# vector_db.index(chunks, embeddings)
```

## Example 6: Custom Response Parser

For non-JSON outputs, use a custom parser.

```python
def parse_key_value(response: str) -> dict:
    """Parse KEY=value format."""
    metadata = {}
    for line in response.split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata

CUSTOM_SYSTEM_PROMPT = """
Extract metadata in KEY=value format (one per line):
CASE_NUMBER=...
COURT=...
DATE=...
"""

extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=CUSTOM_SYSTEM_PROMPT,
    extraction_prompt_template=EXTRACTION_PROMPT,
    response_parser=parse_key_value,
)
```

## Example 7: Batch Processing Multiple Documents

Process multiple documents efficiently using threading.

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def process_single_document(file_path: Path) -> tuple:
    """Process one document and return chunks + embeddings."""
    # Parse
    parsed = ingestion.parse_file(str(file_path))
    full_text = " ".join([p["text"] for p in parsed])
    
    # Extract metadata
    metadata = extractor.extract(full_text)
    
    # Chunk
    structured = dynamic_chunker.build_chunks(parsed)
    chunks = token_chunker.chunk(structured)
    
    # Add metadata
    for chunk in chunks:
        chunk.metadata.update(metadata)
    
    # Enrich and embed
    enriched = enricher.enrich_chunks(chunks)
    embeddings = embed_client.embed_batch(enriched)
    
    return chunks, embeddings

# Process multiple files in parallel
document_files = list(Path("documents/").glob("*.pdf"))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_single_document, document_files))

# Index all results
for chunks, embeddings in results:
    vector_db.index(chunks, embeddings)

print(f"Processed and indexed {len(results)} documents")
```

## Running the Examples

All examples are available in the repository:

```bash
# Clone repository
git clone https://github.com/gmottola00/quaerium.git
cd quaerium

# Install with examples
pip install -e ".[dev,all]"

# Run metadata extraction example
python examples/metadata_extraction_example.py

# Or use Make command
make run-metadata-example
```

## Next Steps

- 📚 **[User Guide](../guides/metadata_extraction.md)**: Detailed concepts and best practices
- 🔧 **[API Reference](../api/core/metadata.md)**: Complete API documentation
- 🚀 **[RAG Pipeline](../guides/rag_pipeline.md)**: Integrate into full RAG workflow
