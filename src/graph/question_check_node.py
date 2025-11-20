"""
LLM Input Safety Check Module

"""

from typing import Any, Dict

import torch
import torch._inductor.config
from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

from src.graph.state import AgentState

torch.set_float32_matmul_precision("high")
torch._inductor.config.fx_graph_cache = True


