from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from llm_transparency_tool.server.utils import get_contribution_graph
from llm_transparency_tool.routes.graph import build_paths_to_predictions



def nx_graph_to_json(g, max_edges: int | None = None):
    edges = []
    for u, v, data in g.edges(data=True):
        edges.append({
            "src": str(u),
            "dst": str(v),
            "weight": float(data.get("weight", 0.0)),
        })
    # optional: cap for rendering perf (does NOT change algorithm, only display)
    if max_edges is not None and len(edges) > max_edges:
        edges = sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)[:max_edges]
    return {"nodes": [str(n) for n in g.nodes()], "edges": edges}




@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number"
    default: Any = None
    options: Optional[List[str]] = None
    help: str = ""

class MetaTransparencyGraph:
    id = "meta_transparency_graph"
    name = "Information Flow Routes (LLM Transparency Tool)"

    def __init__(self):
        self._model = None
        self._current_model_name = None

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec("model_name", "Model", "select", options=["facebook/opt-125m", "gpt2-small"], default="facebook/opt-125m"),
            FieldSpec("text", "Input Prompt", "textarea", default="The Eiffel Tower is in"),
            FieldSpec("threshold", "Threshold for pruning", "number", default=0.04, help="Lower = less pruning, and hence more complex paths.")
        ]
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = inputs["model_name"]
        text = inputs["text"]
        threshold = float(inputs["threshold"])

        if self._model is None or self._current_model_name != model_name:
            device = "gpu" if torch.cuda.is_available() else "cpu"  # LM-TT uses "gpu"/"cpu" :contentReference[oaicite:4]{index=4}
            self._model = TransformerLensTransparentLlm(model_name=model_name, device=device)
            self._current_model_name = model_name

        with torch.inference_mode():
            self._model.run([text])

        toks_tensor = self._model.tokens()[0]                 # token ids tensor
        token_ids = toks_tensor.tolist()
        token_strs = self._model.tokens_to_strings(toks_tensor)

        raw_graph = get_contribution_graph(
            model=self._model,
            model_key="session_graph",
            tokens=token_strs,        # for cache key
            threshold=threshold
        )

        n_layers = self._model.model_info().n_layers
        n_tokens = len(token_ids)

        focus_token = n_tokens - 1  # or your selected index


        graph_data_nx = build_paths_to_predictions(
            raw_graph,
            n_layers,
            n_tokens,
            [focus_token],  
            threshold
        )

        def _first_graph(obj):
            if hasattr(obj, "edges") and hasattr(obj, "nodes"):
                return obj
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                return _first_graph(obj[0])
            raise TypeError(f"Unexpected build_paths_to_predictions output type: {type(obj)}")

        g_focus = _first_graph(graph_data_nx)


        graph_data = nx_graph_to_json(g_focus, max_edges=int(inputs.get("max_edges", 300)))

        #graph_data = nx_graph_to_json(graph_data_nx, max_edges=1000)  # cap for rendering perf 


        return {
            "plugin": self.id,
            "model": model_name,
            "threshold": threshold,
            "focus_token_index": focus_token,
            "graph_data": graph_data,   # dict
            "tokens": token_strs,
            "model_info": self._model.model_info(),
        }
