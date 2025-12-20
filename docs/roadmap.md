# Roadmap

The future development plan for rag-toolkit.

## Current Status: v0.1.0

✅ Core architecture with Protocol-based design  
✅ Ollama and OpenAI integrations  
✅ Milvus vector store  
✅ Basic RAG pipeline  
✅ Documentation and examples  

## Version 0.2.0 (Q1 2025)

**Focus**: Multi-VectorStore Support

### Planned Features

**Vector Stores**
- [ ] Pinecone integration
- [ ] Qdrant integration
- [ ] ChromaDB integration
- [ ] Weaviate integration (community contribution)

**Improvements**
- [ ] Unified vector store testing suite
- [ ] Performance benchmarks across stores
- [ ] Migration tools between vector stores
- [ ] Vector store selection guide

**Documentation**
- [ ] Vector store comparison guide
- [ ] Migration tutorials
- [ ] Performance optimization guide

## Version 0.3.0 (Q2 2025)

**Focus**: Enhanced RAG Techniques

### Planned Features

**Retrieval Enhancements**
- [ ] Cross-encoder reranking (Cohere, Jina)
- [ ] Reciprocal Rank Fusion (RRF)
- [ ] Query decomposition
- [ ] Multi-query retrieval with fusion
- [ ] Parent-child document retrieval

**Chunking Strategies**
- [ ] Semantic chunking
- [ ] Sliding window with overlap
- [ ] Hierarchical chunking
- [ ] Custom chunking callbacks

**Context Management**
- [ ] Smart context pruning
- [ ] Context caching
- [ ] Conversation memory
- [ ] Multi-turn dialogue support

**Documentation**
- [ ] Advanced RAG techniques guide
- [ ] Reranking strategies comparison
- [ ] Context management best practices

## Version 0.4.0 (Q3 2025)

**Focus**: Production Features

### Planned Features

**Observability**
- [ ] Built-in tracing with OpenTelemetry
- [ ] Performance metrics
- [ ] Cost tracking
- [ ] Query analytics dashboard

**Optimization**
- [ ] Response caching
- [ ] Batch processing optimization
- [ ] Streaming responses
- [ ] GPU acceleration support

**Evaluation**
- [ ] Built-in evaluation framework
- [ ] Retrieval metrics (Precision@K, MRR, NDCG)
- [ ] Generation metrics (BLEU, ROUGE, BERTScore)
- [ ] A/B testing support

**Deployment**
- [ ] Docker compose templates
- [ ] Kubernetes manifests
- [ ] Serverless deployment guides
- [ ] Load balancing strategies

**Documentation**
- [ ] Production deployment guide
- [ ] Monitoring and observability
- [ ] Performance tuning
- [ ] Cost optimization

## Version 0.5.0 (Q4 2025)

**Focus**: Advanced Features

### Planned Features

**Multi-Modal RAG**
- [ ] Image embeddings (CLIP)
- [ ] Document image processing
- [ ] Multi-modal retrieval
- [ ] Vision-language models integration

**Graph RAG**
- [ ] Knowledge graph integration
- [ ] Entity extraction
- [ ] Relationship mapping
- [ ] Graph-based retrieval

**Advanced LLM Features**
- [ ] Function calling support
- [ ] Tool use integration
- [ ] Agent-based RAG
- [ ] Self-querying retrieval

**Security**
- [ ] API key management
- [ ] Rate limiting
- [ ] User authentication
- [ ] Data privacy controls

## Long-term Vision

### Community Features
- Plugin system for custom components
- Community vector store adapters
- Shared embedding models
- Pre-built RAG templates

### Enterprise Features
- Multi-tenancy support
- RBAC (Role-Based Access Control)
- Audit logging
- Compliance tools (GDPR, SOC2)

### Research Integration
- Latest RAG techniques from papers
- Experimental features branch
- Research partnership program

## How to Contribute

We welcome contributions! Here's how you can help:

### Priority Areas

1. **Vector Store Implementations**
   - Implement new vector store adapters
   - See `src/rag_toolkit/core/vectorstore.py` for the protocol

2. **LLM Provider Integrations**
   - Add support for new LLM providers
   - See `src/rag_toolkit/core/llm.py` for the protocol

3. **Embedding Models**
   - Integrate new embedding models
   - See `src/rag_toolkit/core/embedding.py` for the protocol

4. **Examples and Tutorials**
   - Write tutorials for specific use cases
   - Share your RAG applications
   - Add to `examples/` directory

5. **Documentation**
   - Improve existing docs
   - Add missing sections
   - Translate to other languages

### Feature Requests

Have an idea? [Open an issue](https://github.com/gmottola00/rag-toolkit/issues/new) with:
- Clear description of the feature
- Use case and benefits
- Proposed implementation (optional)
- Willingness to contribute

### Voting

Check our [GitHub Discussions](https://github.com/gmottola00/rag-toolkit/discussions) to:
- Vote on proposed features
- Suggest new features
- Share your use cases

## Release Schedule

- **Minor versions** (0.x.0): Quarterly
- **Patch versions** (0.x.y): As needed for bug fixes
- **Major version** (1.0.0): When API is stable and battle-tested

## Stability Guarantees

### Current (0.x.x)
- ⚠️ API may change between minor versions
- ✅ We'll provide migration guides
- ✅ Deprecation warnings before breaking changes

### Future (1.0.0+)
- ✅ Semantic versioning
- ✅ Stable API with deprecation cycles
- ✅ LTS (Long Term Support) releases

## Stay Updated

- 📧 [Mailing list](https://github.com/gmottola00/rag-toolkit) (coming soon)
- 🐦 [Twitter](https://twitter.com) (coming soon)
- 💬 [Discord community](https://discord.gg) (coming soon)
- ⭐ [Star on GitHub](https://github.com/gmottola00/rag-toolkit)

---

*This roadmap is subject to change based on community feedback and priorities.*

Last updated: December 20, 2024
