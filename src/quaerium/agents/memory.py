"""Memory implementations for Quaerium Agents.

Provides:
- InMemoryConversationMemory — sliding window, zero dependencies
- VectorLongTermMemory — semantic recall using existing VectorStoreClient
"""

from __future__ import annotations

from collections import deque
from typing import Any


class InMemoryConversationMemory:
    """Simple sliding-window conversation memory.

    Satisfies the AgentMemory Protocol. Zero external dependencies.

    Args:
        max_turns: Maximum number of (user, assistant) turn pairs to keep.
            None means unlimited.
    """

    def __init__(self, max_turns: int | None = 50) -> None:
        self._max_turns = max_turns
        self._messages: deque[dict[str, str]] = deque()

    def add_message(self, role: str, content: str) -> None:
        """Append a message and enforce the sliding window."""
        self._messages.append({"role": role, "content": content})
        if self._max_turns is not None:
            # Each turn = 2 messages (user + assistant), keep max_turns * 2
            max_messages = self._max_turns * 2
            while len(self._messages) > max_messages:
                self._messages.popleft()

    def get_history(self, max_turns: int | None = None) -> list[dict[str, str]]:
        """Return conversation history, optionally limited to last N turns."""
        messages = list(self._messages)
        if max_turns is not None:
            messages = messages[-(max_turns * 2):]
        return messages

    def clear(self) -> None:
        """Clear all stored messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class VectorLongTermMemory:
    """Long-term memory backed by a vector store.

    Provides semantic recall on top of any VectorStoreClient +
    EmbeddingClient. This class is NOT part of the AgentMemory Protocol
    (it has extra methods); use composition, not polymorphism.

    Args:
        vector_store: VectorStoreClient instance
        embedding_client: EmbeddingClient instance
        collection_name: Collection/namespace for storing memories
        top_k: Number of relevant memories to recall
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_client: Any,
        collection_name: str = "agent_memory",
        top_k: int = 5,
    ) -> None:
        self._store = vector_store
        self._embedding = embedding_client
        self._collection = collection_name
        self._top_k = top_k
        # Short-term buffer that satisfies AgentMemory Protocol
        self._short_term = InMemoryConversationMemory()

    # --- AgentMemory Protocol methods (short-term) ---

    def add_message(self, role: str, content: str) -> None:
        self._short_term.add_message(role, content)

    def get_history(self, max_turns: int | None = None) -> list[dict[str, str]]:
        return self._short_term.get_history(max_turns)

    def clear(self) -> None:
        self._short_term.clear()

    # --- Extra long-term memory methods ---

    def remember(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a piece of information in long-term vector memory."""
        vector = self._embedding.embed(text)
        self._store.upsert(
            collection=self._collection,
            vectors=[{"id": _hash_text(text), "vector": vector, "text": text, "metadata": metadata or {}}],
        )

    def recall(self, query: str) -> list[str]:
        """Retrieve semantically relevant memories for a query."""
        vector = self._embedding.embed(query)
        results = self._store.search(
            collection=self._collection,
            vector=vector,
            top_k=self._top_k,
        )
        return [r.get("text", "") for r in results]


def _hash_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


__all__ = ["InMemoryConversationMemory", "VectorLongTermMemory"]
