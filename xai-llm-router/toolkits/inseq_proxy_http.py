from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import requests

@dataclass
class FieldSpec:
    key: str
    label: str
    type: str
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None

class ToolkitPlugin:
    id: str
    name: str
    def spec(self) -> List[FieldSpec]: ...
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

class _InseqHTTPBase(ToolkitPlugin):
    base_url: str = os.environ.get("INSEQ_URL", "http://127.0.0.1:8001")
    remote_id: str = ""

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/run",
            json={"plugin": self.remote_id, "inputs": inputs},
            timeout=600,
        )
        if not r.ok:
            raise RuntimeError(f"Inseq service error ({r.status_code}): {r.text}")
        return r.json()

class InseqDecoderIG_HTTP(_InseqHTTPBase):
    id = "inseq_decoder_ig"
    remote_id = "inseq_decoder_ig"
    name = "Integrated Gradients (Inseq) [service]"

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec("model_name", "HF model name (decoder)", "text",
                      help="Example: gpt2, arnir0/Tiny-LLM"),
            FieldSpec("sentence", "Input sentence", "textarea"),
            FieldSpec("max_new_tokens", "Max new tokens", "number", default=5),
            FieldSpec("n_steps", "n_steps", "number", default=100),
            FieldSpec("internal_batch_size", "internal_batch_size", "number", default=50),
        ]

class InseqEncDecIG_HTTP(_InseqHTTPBase):
    id = "inseq_encdec_ig"
    remote_id = "inseq_encdec_ig"
    name = "Integrated Gradients (Inseq Enc-Dec) [service]"

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec("model_name", "HF model name (enc-dec)", "text",
                      help="Example: puettmann/Foglietta-mt-en-it, Helsinki-NLP/opus-mt-en-fr"),
            FieldSpec("sentence", "Input sentence", "textarea"),
            FieldSpec("n_steps", "n_steps", "number", default=100),
        ]
