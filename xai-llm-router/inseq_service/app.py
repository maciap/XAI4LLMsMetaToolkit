# inseq_service/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Type

app = FastAPI(title="Inseq Service")

class RunReq(BaseModel):
    plugin: str
    inputs: Dict[str, Any]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/run")
def run(req: RunReq):
    try:
        from toolkits.inseq import (
            InseqDecoderIG, InseqEncDecIG,
            InseqDecoderGradientSHAP, InseqEncDecGradientSHAP,
            InseqDecoderDeepLIFT, InseqEncDecDeepLIFT,
            InseqDecoderInputXGradient, InseqEncDecInputXGradient,
            InseqDecoderLIME, InseqEncDecLIME,
            InseqDecoderDiscretizedIG, InseqEncDecDiscretizedIG
        )

        REGISTRY: Dict[str, Type] = {
            "inseq_decoder_ig": InseqDecoderIG,
            "inseq_encdec_ig": InseqEncDecIG,
            "inseq_decoder_gradient_shap": InseqDecoderGradientSHAP,
            "inseq_encdec_gradient_shap": InseqEncDecGradientSHAP,
            "inseq_decoder_deeplift": InseqDecoderDeepLIFT,
            "inseq_encdec_deeplift": InseqEncDecDeepLIFT,
            "inseq_decoder_input_x_gradient": InseqDecoderInputXGradient,
            "inseq_encdec_input_x_gradient": InseqEncDecInputXGradient,
            "inseq_decoder_lime": InseqDecoderLIME,
            "inseq_encdec_lime": InseqEncDecLIME,
            "inseq_decoder_discretized_ig": InseqDecoderDiscretizedIG,
            "inseq_encdec_discretized_ig": InseqEncDecDiscretizedIG,
        }

        cls = REGISTRY.get(req.plugin)
        if cls is None:
            raise ValueError(f"Unknown plugin: {req.plugin}. Known: {sorted(REGISTRY.keys())}")

        return cls().run(req.inputs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))