from typing import Any, Dict

import torch
import torch._inductor.config
from llm_guard import scan_output
from llm_guard.output_scanners import LanguageSame, Relevance, Sentiment

from src.graph.state import AgentState

torch.set_float32_matmul_precision("high")
torch._inductor.config.fx_graph_cache = True

language_same_scanner = LanguageSame(use_onnx=True)
relevance_scanner = Relevance(use_onnx=True)
sentiment_scanner = Sentiment()


