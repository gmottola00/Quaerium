"""Example: Extract metadata from legal documents and enrich chunks.

This example demonstrates:
1. LLM-based metadata extraction from documents
2. Chunking with DynamicChunker and TokenChunker
3. Metadata enrichment for better retrieval
4. Batch embedding of enriched chunks

Domain: Legal documents (contracts, court filings)
"""

from quaerium.core.metadata import LLMMetadataExtractor
from quaerium.core.chunking import DynamicChunker, TokenChunker, MetadataEnricher
from quaerium.infra.llm import OllamaLLMClient
from quaerium.infra.embedding import OllamaEmbeddingClient


# ============================================================================
# Step 1: Define Domain-Specific Extraction Prompts
# ============================================================================

LEGAL_SYSTEM_PROMPT = """
You are a legal document analyzer. Your task is to extract structured metadata
from legal documents and return it in the following JSON format:

{
  "case_number": "",
  "court": "",
  "filing_date": "",
  "parties": [],
  "document_type": ""
}

Rules:
- If a field is not found, use empty string ("") or empty array ([])
- Return ONLY valid JSON, no additional text
- Be precise with extracted values
"""

LEGAL_EXTRACTION_PROMPT = """
Given the following legal document excerpt:

{context}

Extract the following information:
- case_number: The case identifier (e.g., "2024-CV-12345")
- court: The name of the court (e.g., "Superior Court of California")
- filing_date: The filing or effective date
- parties: List of party names (plaintiffs, defendants)
- document_type: Type of document (e.g., "complaint", "motion", "contract")

Return ONLY the JSON object with extracted metadata.
"""


# ============================================================================
# Step 2: Sample Document
# ============================================================================

SAMPLE_LEGAL_DOCUMENT = """
SUPERIOR COURT OF CALIFORNIA
COUNTY OF LOS ANGELES

Case No. 2024-CV-12345

John Doe, Plaintiff
v.
Acme Corporation, Defendant

Filed: January 15, 2024

COMPLAINT FOR BREACH OF CONTRACT

This case concerns a contract dispute between the parties. The Plaintiff,
John Doe, entered into a written employment agreement with Defendant
Acme Corporation on March 1, 2023. The contract stipulated a two-year
term with specific performance milestones and compensation structure.

Defendant breached the contract by failing to provide the agreed-upon
resources and subsequently terminating Plaintiff's employment without
cause on November 30, 2023. This breach caused Plaintiff significant
financial and professional harm.

Plaintiff seeks damages in the amount of $250,000 for lost wages,
benefits, and reputational damage, plus attorney's fees and costs.
"""


# ============================================================================
# Main Example
# ============================================================================

def main():
    print("="*80)
    print("METADATA EXTRACTION & ENRICHMENT EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # Initialize Components
    # ========================================================================
    
    print("\n1️⃣  Initializing components...")
    
    # LLM for metadata extraction
    llm = OllamaLLMClient(model="phi3:mini")
    print(f"   ✓ LLM client: {llm.model_name}")
    
    # Embedding for vector indexing
    embed_client = OllamaEmbeddingClient(model="nomic-embed-text")
    print(f"   ✓ Embedding client: {embed_client.model_name}")
    
    # Metadata extractor
    metadata_extractor = LLMMetadataExtractor(
        llm_client=llm,
        system_prompt=LEGAL_SYSTEM_PROMPT,
        extraction_prompt_template=LEGAL_EXTRACTION_PROMPT,
        max_text_length=8000,
    )
    print("   ✓ Metadata extractor configured")
    
    # Chunkers
    token_chunker = TokenChunker(max_tokens=512, overlap_tokens=50)
    print("   ✓ Token chunker (max=512, overlap=50)")
    
    # Metadata enricher
    enricher = MetadataEnricher(
        format_template="[{key}: {value}]",
        excluded_keys=["file_name", "chunk_id", "id"],
    )
    print("   ✓ Metadata enricher configured")
    
    # ========================================================================
    # Extract Metadata
    # ========================================================================
    
    print("\n2️⃣  Extracting metadata from document...")
    print(f"   Document length: {len(SAMPLE_LEGAL_DOCUMENT)} characters")
    
    document_metadata = metadata_extractor.extract(SAMPLE_LEGAL_DOCUMENT)
    
    print("\n   📋 Extracted metadata:")
    for key, value in document_metadata.items():
        if isinstance(value, list):
            print(f"      • {key}: {', '.join(value) if value else '(none)'}")
        else:
            print(f"      • {key}: {value or '(none)'}")
    
    # ========================================================================
    # Create Mock Chunks (In production, use DynamicChunker with parsed docs)
    # ========================================================================
    
    print("\n3️⃣  Creating document chunks...")
    
    # For this example, we'll manually create chunks from the document
    # In production, you would use:
    # - PDF/DOCX parser to get structured pages
    # - DynamicChunker to create semantic chunks
    # - TokenChunker to optimize for embedding
    
    from dataclasses import dataclass, field
    from typing import Dict, Any, List
    
    @dataclass
    class MockTokenChunk:
        """Mock chunk for example."""
        id: str
        text: str
        section_path: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        page_numbers: List[int] = field(default_factory=list)
        source_chunk_id: str = ""
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "id": self.id,
                "text": self.text,
                "metadata": self.metadata,
            }
    
    # Split document into logical chunks
    paragraphs = [p.strip() for p in SAMPLE_LEGAL_DOCUMENT.split('\n\n') if p.strip()]
    
    chunks = []
    for i, para in enumerate(paragraphs):
        chunk = MockTokenChunk(
            id=f"chunk-{i+1}",
            text=para,
            section_path="Legal Document > Body",
            metadata=document_metadata.copy(),
            page_numbers=[1],
            source_chunk_id=f"source-{i+1}",
        )
        chunks.append(chunk)
    
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # ========================================================================
    # Enrich Chunks
    # ========================================================================
    
    print("\n4️⃣  Enriching chunk text with metadata...")
    
    enriched_texts = enricher.enrich_chunks(chunks)
    
    print(f"   ✓ Enriched {len(enriched_texts)} chunks")
    
    # Show example
    print("\n   📝 Example enriched chunk:")
    print(f"      Original: {chunks[3].text[:80]}...")
    print(f"      Enriched: {enriched_texts[3][:150]}...")
    
    # ========================================================================
    # Generate Embeddings
    # ========================================================================
    
    print("\n5️⃣  Generating embeddings for enriched chunks...")
    
    embeddings = embed_client.embed_batch(enriched_texts)
    
    print(f"   ✓ Generated {len(embeddings)} embeddings")
    print(f"   ✓ Embedding dimension: {len(embeddings[0])}")
    
    # ========================================================================
    # Summary & Next Steps
    # ========================================================================
    
    print("\n6️⃣  Summary:")
    print(f"   • Document length: {len(SAMPLE_LEGAL_DOCUMENT)} chars")
    print(f"   • Metadata fields extracted: {len(document_metadata)}")
    print(f"   • Chunks created: {len(chunks)}")
    print(f"   • Enriched chunks: {len(enriched_texts)}")
    print(f"   • Embeddings generated: {len(embeddings)}")
    
    print("\n7️⃣  Next Steps (In Production):")
    print("   1. Store chunks in vector database (Milvus/Chroma/Qdrant)")
    print("   2. Index with enriched text embeddings")
    print("   3. Use metadata for filtering during retrieval")
    print("   4. Implement hybrid search (vector + metadata filters)")
    
    print("\n" + "="*80)
    print("✅ Example completed successfully!")
    print("="*80)
    
    # Return results for potential further use
    return {
        "metadata": document_metadata,
        "chunks": chunks,
        "enriched_texts": enriched_texts,
        "embeddings": embeddings,
    }


if __name__ == "__main__":
    # Run example
    # Note: Requires Ollama to be running with phi3:mini and nomic-embed-text models
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   • Ollama is running (ollama serve)")
        print("   • Models are pulled:")
        print("     - ollama pull phi3:mini")
        print("     - ollama pull nomic-embed-text")
