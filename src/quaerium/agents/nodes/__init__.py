"""Graph nodes: router, model, gate, act. Built by :func:`quaerium.agents.graph.build_graph`."""

from __future__ import annotations

from quaerium.agents.nodes.act import make_act
from quaerium.agents.nodes.gate import make_gate
from quaerium.agents.nodes.model import after_model, make_model_node
from quaerium.agents.nodes.router import make_router

__all__ = ["after_model", "make_act", "make_gate", "make_model_node", "make_router"]
