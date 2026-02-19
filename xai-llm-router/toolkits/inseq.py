from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import inseq


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None

class ToolkitPlugin:
    id: str
    name: str

    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    
class InseqDecoderIG(ToolkitPlugin):
    id = "inseq_decoder_ig"
    name = "Integrated Gradients Decoder (Inseq)"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cache: tokenizer/model by model name
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key = "model_name",
                label = "HF model name (decoder)",
                type = "text",
                help = "Example: gpt2, arnir0/Tiny-LLM"
            ),
            FieldSpec(
                key="sentence",
                label="Input sentence",
                type="textarea",
                help=""
            ),
            FieldSpec(
                key="max_new_tokens",
                label="Max new tokens",
                type="number",
                help="",
                default=5
            ),
            FieldSpec(
                key="n_steps",
                label="n_steps",
                type="number",
                help="",
                default=100
            ),
            FieldSpec(
                key="internal_batch_size",
                label="internal_batch_size",
                type="number",
                help="",
                default=50
            )
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "arnir0/Tiny-LLM"
        text = (inputs.get("sentence") or "").strip()
        if not text:
            raise ValueError("Text is empty.")
        
        max_new_tokens = int(inputs.get("max_new_tokens"))
        n_steps = int(inputs.get("n_steps") or "")
        internal_batch_size = int(inputs.get("internal_batch_size"))

        if model_name == "arnir0/Tiny-LLM":
            model = inseq.load_model(model_name, "integrated_gradients", generation_args={"max_new_tokens": max_new_tokens}, n_steps=n_steps, internal_batch_size=internal_batch_size,
                                     tokenizer_kwargs={"add_bos_token": False, "pad_token_id": 0})
        else:
            model = inseq.load_model(model_name, "integrated_gradients")
        out = model.attribute(text, generation_args={"max_new_tokens": max_new_tokens}, n_steps=n_steps, internal_batch_size=internal_batch_size)
        # out.show()

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "out": out.show(display=False, return_html=True)
        }
    
class InseqEncDecIG(ToolkitPlugin):
    id = "inseq_encdec_ig"
    name = "Integrated Gradients EncoderDecoder (Inseq)"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cache: tokenizer/model by model name
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key = "model_name",
                label = "HF model name (decoder)",
                type = "text",
                help = "Example: puettmann/Foglietta-mt-en-it, Helsinki-NLP/opus-mt-en-fr"
            ),
            FieldSpec(
                key="sentence",
                label="Input sentence",
                type="textarea",
                help="",
            ),
            FieldSpec(
                key="n_steps",
                label="n_steps",
                type="number",
                help="",
                default=100,
            ),
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "puettmann/Foglietta-mt-en-it"
        text = (inputs.get("sentence") or "").strip()
        if not text:
            raise ValueError("Text is empty.")

        n_steps = int(inputs.get("n_steps"))

        model = inseq.load_model(model_name, "integrated_gradients")
        out = model.attribute(text, n_steps=n_steps)
        # out.show()

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "out": out.show(display=False, return_html=True)
        }
