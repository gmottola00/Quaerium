# :material-tune-vertical: Advanced RAG Pipeline

!!! success "Production-Ready Features"
    Learn how to build a production-ready RAG pipeline with advanced features like query rewriting, reranking, conversational memory, and multi-query strategies.

---

## :material-rocket: Complete Advanced Pipeline

### :material-star: Full-Featured Implementation

!!! example "All Features Enabled"
    This implementation includes all advanced RAG capabilities.

```python title="advanced_pipeline.py" linenums="1" hl_lines="9-19 34-36 39"
from rag_toolkit import RagPipeline
from rag_toolkit.infra.embedding import OpenAIEmbedding
from rag_toolkit.infra.llm import OpenAILLM
from rag_toolkit.core.vectorstore import MilvusVectorStore
from rag_toolkit.core.reranker import CrossEncoderReranker
from rag_toolkit.core.types import SearchResult
import asyncio

class AdvancedRagPipeline:
    """Advanced RAG pipeline with all features."""
    
    def __init__(
        self,
        embedding_client: OpenAIEmbedding,
        vector_store: MilvusVectorStore,
        llm_client: OpenAILLM,
        reranker: CrossEncoderReranker | None = None,
        enable_query_rewriting: bool = True,
        enable_multi_query: bool = False,
        enable_conversation: bool = True,
    ):
        self.embedding = embedding_client
        self.vector_store = vector_store
        self.llm = llm_client
        self.reranker = reranker
        
        self.enable_query_rewriting = enable_query_rewriting
        self.enable_multi_query = enable_multi_query
        self.enable_conversation = enable_conversation
        
        # Conversation history
        self.conversation_history: list[dict] = []
    
    async def query(
        self,
        query: str,
        limit: int = 5,
        filter: dict | None = None,
    ):
        """Advanced query with all features."""
        # 1. Query rewriting
        if self.enable_query_rewriting:
            rewritten_query = await self._rewrite_query(query)
        else:
            rewritten_query = query
        
        # 2. Multi-query generation
        if self.enable_multi_query:
            queries = await self._generate_multi_queries(rewritten_query)
        else:
            queries = [rewritten_query]
        
        # 3. Retrieve documents
        all_results = await self._retrieve_multi_query(
            queries=queries,
            limit=limit * 2,
            filter=filter
        )
        
        # 4. Rerank
        if self.reranker:
            reranked = await self.reranker.rerank(
                query=query,
                documents=all_results,
                top_k=limit
            )
        else:
            reranked = all_results[:limit]
        
        # 5. Generate answer
        answer = await self._generate_answer(
            query=query,
            context=reranked
        )
        
        # 6. Update conversation history
        if self.enable_conversation:
            self._update_history(query, answer)
        
        return {
            "answer": answer,
            "sources": reranked,
            "original_query": query,
            "rewritten_query": rewritten_query
        }

# Usage
pipeline = AdvancedRagPipeline(
    embedding_client=OpenAIEmbedding(),
    vector_store=MilvusVectorStore(
        collection_name="docs",
        dimension=1536
    ),
    llm_client=OpenAILLM(model="gpt-4-turbo"),
    reranker=CrossEncoderReranker(),
    enable_query_rewriting=True,
    enable_conversation=True,
)

result = await pipeline.query("What is machine learning?")
print(result["answer"])
```

## Best Practices

1. **Enable Query Rewriting** - Improves retrieval quality
2. **Use Reranking** - Better result ranking
3. **Implement Caching** - Faster repeated queries
4. **Monitor Performance** - Track metrics
5. **Test Thoroughly** - Evaluate on real queries

## Next Steps

- [Production Setup](production_setup.md) - Deploy to production
- [Reranking Guide](../user_guide/reranking.md) - Improve ranking
- [RAG Pipeline Guide](../user_guide/rag_pipeline.md) - Deep dive

