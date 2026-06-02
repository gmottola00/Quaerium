"""Tests for agent memory implementations."""

import pytest

from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.protocols import AgentMemory


class TestInMemoryConversationMemory:
    def test_add_and_get(self):
        mem = InMemoryConversationMemory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi there!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_sliding_window(self):
        mem = InMemoryConversationMemory(max_turns=2)
        for i in range(5):
            mem.add_message("user", f"msg {i}")
            mem.add_message("assistant", f"reply {i}")
        # Only last 2 turns (4 messages) should remain
        history = mem.get_history()
        assert len(history) == 4

    def test_clear(self):
        mem = InMemoryConversationMemory()
        mem.add_message("user", "hello")
        mem.clear()
        assert mem.get_history() == []

    def test_get_history_limited(self):
        mem = InMemoryConversationMemory()
        for i in range(5):
            mem.add_message("user", f"u{i}")
            mem.add_message("assistant", f"a{i}")
        history = mem.get_history(max_turns=2)
        assert len(history) == 4

    def test_unlimited(self):
        mem = InMemoryConversationMemory(max_turns=None)
        for i in range(100):
            mem.add_message("user", f"msg {i}")
        assert len(mem) == 100

    def test_protocol_satisfaction(self):
        mem = InMemoryConversationMemory()
        assert isinstance(mem, AgentMemory)
