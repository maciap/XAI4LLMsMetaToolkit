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
    remote_id: str = ""

    @property
    def base_url(self) -> str:
        # Read env var dynamically (so it works even if env is set after import)
        return os.environ.get("INSEQ_URL", "http://127.0.0.1:8001").rstrip("/")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/run"
        try:
            r = requests.post(
                url,
                json={"plugin": self.remote_id, "inputs": inputs},
                timeout=600,
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not reach Inseq service at {url}. "
                f"Is it running? Error: {e}"
            ) from e

        if not r.ok:
            raise RuntimeError(f"Inseq service error ({r.status_code}): {r.text}")
        return r.json()





# -------------------------
# Shared spec helpers
# -------------------------
def _decoder_common_spec(*, include_n_steps: bool = True, include_internal_bs: bool = False, include_lime_samples: bool = False) -> List[FieldSpec]:
    fields: List[FieldSpec] = [
        FieldSpec("model_name", "HF model name (decoder)", "text",
                  help="Example: gpt2, arnir0/Tiny-LLM"),
        FieldSpec("sentence", "Input sentence", "textarea"),
        FieldSpec("max_new_tokens", "Max new tokens", "number", default=5),
    ]
    if include_n_steps:
        fields.append(FieldSpec("n_steps", "n_steps", "number", default=100))
    if include_internal_bs:
        fields.append(FieldSpec("internal_batch_size", "internal_batch_size", "number", default=50))
    if include_lime_samples:
        fields.append(FieldSpec("num_samples", "num_samples (LIME)", "number", default=200))
    return fields

def _encdec_common_spec(*, include_n_steps: bool = True, include_lime_samples: bool = False) -> List[FieldSpec]:
    fields: List[FieldSpec] = [
        FieldSpec("model_name", "HF model name (enc-dec)", "text",
                  help="Example: puettmann/Foglietta-mt-en-it, Helsinki-NLP/opus-mt-en-fr"),
        FieldSpec("sentence", "Input sentence", "textarea"),
    ]
    if include_n_steps:
        fields.append(FieldSpec("n_steps", "n_steps", "number", default=100))
    if include_lime_samples:
        fields.append(FieldSpec("num_samples", "num_samples (LIME)", "number", default=200))
    return fields


# -------------------------
# Existing IG proxies
# -------------------------
class InseqDecoderIG_HTTP(_InseqHTTPBase):
    id = "inseq_decoder_ig"
    remote_id = "inseq_decoder_ig"
    name = "Integrated Gradients (Inseq) [service]"

    def spec(self) -> List[FieldSpec]:
        return _decoder_common_spec(include_n_steps=True, include_internal_bs=True)

class InseqEncDecIG_HTTP(_InseqHTTPBase):
    id = "inseq_encdec_ig"
    remote_id = "inseq_encdec_ig"
    name = "Integrated Gradients (Inseq Enc-Dec) [service]"

    def spec(self) -> List[FieldSpec]:
        return _encdec_common_spec(include_n_steps=True)


# -------------------------
# GradientSHAP proxies
# -------------------------
class InseqDecoderGradientSHAP_HTTP(_InseqHTTPBase):
    id = "inseq_decoder_gradient_shap"
    remote_id = "inseq_decoder_gradient_shap"
    name = "GradientSHAP (Inseq) [service]"

    def spec(self) -> List[FieldSpec]:
        # GradientSHAP usually supports n_steps and benefits from batching
        return _decoder_common_spec(include_n_steps=True, include_internal_bs=True)

class InseqEncDecGradientSHAP_HTTP(_InseqHTTPBase):
    id = "inseq_encdec_gradient_shap"
    remote_id = "inseq_encdec_gradient_shap"
    name = "GradientSHAP (Inseq Enc-Dec) [service]"

    def spec(self) -> List[FieldSpec]:
        return _encdec_common_spec(include_n_steps=True)


# -------------------------
# DeepLIFT proxies
# -------------------------
class InseqDecoderDeepLIFT_HTTP(_InseqHTTPBase):
    id = "inseq_decoder_deeplift"
    remote_id = "inseq_decoder_deeplift"
    name = "DeepLIFT (Inseq) [service]"

    def spec(self) -> List[FieldSpec]:
        # Some implementations use n_steps; if your inseq DeepLIFT doesn't, you can set include_n_steps=False
        return _decoder_common_spec(include_n_steps=True, include_internal_bs=False)

class InseqEncDecDeepLIFT_HTTP(_InseqHTTPBase):
    id = "inseq_encdec_deeplift"
    remote_id = "inseq_encdec_deeplift"
    name = "DeepLIFT (Inseq Enc-Dec) [service]"

    def spec(self) -> List[FieldSpec]:
        return _encdec_common_spec(include_n_steps=True)


# -------------------------
# Input × Gradient proxies
# -------------------------
class InseqDecoderInputXGradient_HTTP(_InseqHTTPBase):
    id = "inseq_decoder_input_x_gradient"
    remote_id = "inseq_decoder_input_x_gradient"
    name = "Input×Gradient (Inseq) [service]"

    def spec(self) -> List[FieldSpec]:
        # Input×Grad is typically single-pass; n_steps not needed
        return _decoder_common_spec(include_n_steps=False, include_internal_bs=False)

class InseqEncDecInputXGradient_HTTP(_InseqHTTPBase):
    id = "inseq_encdec_input_x_gradient"
    remote_id = "inseq_encdec_input_x_gradient"
    name = "Input×Gradient (Inseq Enc-Dec) [service]"

    def spec(self) -> List[FieldSpec]:
        return _encdec_common_spec(include_n_steps=False)


# -------------------------
# LIME proxies
# -------------------------
class InseqDecoderLIME_HTTP(_InseqHTTPBase):
    id = "inseq_decoder_lime"
    remote_id = "inseq_decoder_lime"
    name = "LIME (Inseq) [service]"

    def spec(self) -> List[FieldSpec]:
        # LIME is perturbation-based; include num_samples instead of n_steps/internal_batch_size
        return _decoder_common_spec(include_n_steps=False, include_internal_bs=False, include_lime_samples=True)

class InseqEncDecLIME_HTTP(_InseqHTTPBase):
    id = "inseq_encdec_lime"
    remote_id = "inseq_encdec_lime"
    name = "LIME (Inseq Enc-Dec) [service]"

    def spec(self) -> List[FieldSpec]:
        return _encdec_common_spec(include_n_steps=False, include_lime_samples=True)
    

class InseqDecoderDiscretizedIG_HTTP(_InseqHTTPBase):
    id = "inseq_decoder_discretized_ig"
    remote_id = "inseq_decoder_discretized_ig"
    name = "Discretized IG (Inseq) [service]"

    def spec(self) -> List[FieldSpec]:
        return _decoder_common_spec(include_n_steps=True, include_internal_bs=True)

class InseqEncDecDiscretizedIG_HTTP(_InseqHTTPBase):
    id = "inseq_encdec_discretized_ig"
    remote_id = "inseq_encdec_discretized_ig"
    name = "Discretized IG (Inseq Enc-Dec) [service]"

    def spec(self) -> List[FieldSpec]:
        return _encdec_common_spec(include_n_steps=True)