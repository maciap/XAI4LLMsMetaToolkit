from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

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
        # Import the real plugins only in the xai-inseq env
        from toolkits.inseq import InseqDecoderIG, InseqEncDecIG

        if req.plugin == "inseq_decoder_ig":
            plugin = InseqDecoderIG()
        elif req.plugin == "inseq_encdec_ig":
            plugin = InseqEncDecIG()
        else:
            raise ValueError(f"Unknown plugin: {req.plugin}")

        return plugin.run(req.inputs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
