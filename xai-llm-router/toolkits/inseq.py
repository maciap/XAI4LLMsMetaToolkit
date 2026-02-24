# toolkits/inseq.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import inseq

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


# Map your plugin "method family" to the inseq method identifier
# (If your installed Inseq uses slightly different names, adjust here only.)
METHOD_ID = {
    "ig": "integrated_gradients",
    "discretized_ig": "discretized_integrated_gradients", 
    "gradient_shap": "gradient_shap",
    "deeplift": "deeplift",
    "input_x_gradient": "input_x_gradient",
    "lime": "lime",
}

def _to_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


class _InseqBase(ToolkitPlugin):
    arch: str          # "decoder" | "encdec"
    method_key: str    # keys in METHOD_ID

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Any] = {}  # cache inseq wrappers by (model, method)

    def _spec_common(self) -> List[FieldSpec]:
        if self.arch == "decoder":
            base = [
                FieldSpec("model_name", "HF model name (decoder)", "text", help="Example: gpt2, arnir0/Tiny-LLM"),
                FieldSpec("sentence", "Input sentence", "textarea"),
                FieldSpec("max_new_tokens", "Max new tokens", "number", default=5),
            ]
        else:
            base = [
                FieldSpec("model_name", "HF model name (enc-dec)", "text", help="Example: Helsinki-NLP/opus-mt-en-fr"),
                FieldSpec("sentence", "Input sentence", "textarea"),
            ]

        # Most gradient methods accept n_steps
        if self.method_key in ("ig", "gradient_shap", "deeplift"):
            base.append(FieldSpec("n_steps", "n_steps", "number", default=100))

        # Decoder-only batch knob (works for IG-like methods)
        if self.arch == "decoder" and self.method_key in ("ig", "gradient_shap"):
            base.append(FieldSpec("internal_batch_size", "internal_batch_size", "number", default=50))

        # LIME sampling knob (keep minimal)
        if self.method_key == "lime":
            base.append(FieldSpec("num_samples", "num_samples", "number", default=200))

        return base

    def spec(self) -> List[FieldSpec]:
        return self._spec_common()

    def _load(self, model_name: str):
        method_id = METHOD_ID[self.method_key]
        cache_key = f"{model_name}::{method_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if model_name == "arnir0/Tiny-LLM":
            m = inseq.load_model(
                model_name,
                method_id,
                tokenizer_kwargs={"add_bos_token": False, "pad_token_id": 0},
            )
        else:
            m = inseq.load_model(model_name, method_id)

        self._cache[cache_key] = m
        return m

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip()
        text = (inputs.get("sentence") or "").strip()
        if not text:
            raise ValueError("Text is empty.")

        if not model_name:
            model_name = "arnir0/Tiny-LLM" if self.arch == "decoder" else "Helsinki-NLP/opus-mt-en-fr"

        m = self._load(model_name)

        kwargs: Dict[str, Any] = {}

        if "n_steps" in inputs and inputs.get("n_steps") is not None:
            kwargs["n_steps"] = _to_int(inputs.get("n_steps"), 100)

        if "internal_batch_size" in inputs and inputs.get("internal_batch_size") is not None:
            kwargs["internal_batch_size"] = _to_int(inputs.get("internal_batch_size"), 50)

        if "num_samples" in inputs and inputs.get("num_samples") is not None:
            kwargs["num_samples"] = _to_int(inputs.get("num_samples"), 200)

        if self.arch == "decoder":
            max_new_tokens = _to_int(inputs.get("max_new_tokens"), 5)
            out = m.attribute(text, generation_args={"max_new_tokens": max_new_tokens}, **kwargs)
        else:
            out = m.attribute(text, **kwargs)

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "method": METHOD_ID[self.method_key],
            "params": kwargs,
            "out": out.show(display=False, return_html=True),
        }


# --- Existing IDs (keep) ---
class InseqDecoderIG(_InseqBase):
    id = "inseq_decoder_ig"
    name = "Integrated Gradients Decoder (Inseq)"
    arch = "decoder"
    method_key = "ig"

class InseqEncDecIG(_InseqBase):
    id = "inseq_encdec_ig"
    name = "Integrated Gradients EncoderDecoder (Inseq)"
    arch = "encdec"
    method_key = "ig"


# --- New IDs you already have in methods.json ---
class InseqDecoderGradientSHAP(_InseqBase):
    id = "inseq_decoder_gradient_shap"
    name = "GradientSHAP - Decoder (Inseq)"
    arch = "decoder"
    method_key = "gradient_shap"

class InseqEncDecGradientSHAP(_InseqBase):
    id = "inseq_encdec_gradient_shap"
    name = "GradientSHAP - EncoderDecoder (Inseq)"
    arch = "encdec"
    method_key = "gradient_shap"

class InseqDecoderDeepLIFT(_InseqBase):
    id = "inseq_decoder_deeplift"
    name = "DeepLIFT - Decoder (Inseq)"
    arch = "decoder"
    method_key = "deeplift"

class InseqEncDecDeepLIFT(_InseqBase):
    id = "inseq_encdec_deeplift"
    name = "DeepLIFT - EncoderDecoder (Inseq)"
    arch = "encdec"
    method_key = "deeplift"

class InseqDecoderInputXGradient(_InseqBase):
    id = "inseq_decoder_input_x_gradient"
    name = "Input × Gradient - Decoder (Inseq)"
    arch = "decoder"
    method_key = "input_x_gradient"

class InseqEncDecInputXGradient(_InseqBase):
    id = "inseq_encdec_input_x_gradient"
    name = "Input × Gradient - EncoderDecoder (Inseq)"
    arch = "encdec"
    method_key = "input_x_gradient"

class InseqDecoderLIME(_InseqBase):
    id = "inseq_decoder_lime"
    name = "LIME - Decoder (Inseq)"
    arch = "decoder"
    method_key = "lime"

class InseqEncDecLIME(_InseqBase):
    id = "inseq_encdec_lime"
    name = "LIME - EncoderDecoder (Inseq)"
    arch = "encdec"
    method_key = "lime"


class InseqDecoderDiscretizedIG(_InseqBase):
    id = "inseq_decoder_discretized_ig"
    name = "Discretized Integrated Gradients - Decoder (Inseq)"
    arch = "decoder"
    method_key = "discretized_ig"

class InseqEncDecDiscretizedIG(_InseqBase):
    id = "inseq_encdec_discretized_ig"
    name = "Discretized Integrated Gradients - EncoderDecoder (Inseq)"
    arch = "encdec"
    method_key = "discretized_ig"