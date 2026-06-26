import os
import sys
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import traceback

project_root = Path(__file__).resolve().parent.parent.parent  
model_host_dir = project_root / "web_apps" / "utils" / "model_host"
if str(model_host_dir) not in sys.path: sys.path.insert(0, str(model_host_dir))

from lm_server_helper import LlamaServer

app = FastAPI(title="Infrastructure Compute Node Agent Daemon")

class StartNodePayload(BaseModel):
    model_type: str
    port: int
    host: str = "127.0.0.1"
    hf_repo: str
    hf_file: str
    hf_mmproj: str = ""
    n_gpu_layers: int = -1
    ctx_size: int = 2048
    cpu_moe: int = 0
    flash_attn: bool = False
    cache_type_k: str = ""
    cache_type_v: str = ""
    spec_type: str = ""
    spec_draft_n_max: int = 0
    spec_draft_p_min: float = 0.0
    spec_type_secondary: str = ""
    spec_ngram_mod_n_match: int = 0
    jinja: bool = False
    chat_template_kwargs: str = ""
    reasoning_budget: int = 0
    reasoning_budget_message: str = ""
    temperature: float = 0.1
    min_p: float = 0.05
    top_k: int = 40
    top_p: float = 0.95
    presence_penalty: float = 0.0

@app.get("/api/node/status")
async def check_status(port: int, host: str = "127.0.0.1"):
    server = LlamaServer(port=port, host=host)
    return {"online": server.is_port_in_use()}

@app.post("/api/node/start")
async def start_node(payload: StartNodePayload):
    try:
        if not payload.hf_repo or not payload.hf_file:
            raise HTTPException(status_code=400, detail="Missing Hugging Face repo or file name.")

        server = LlamaServer(
            host=payload.host,      
            port=payload.port,
            n_gpu_layers=payload.n_gpu_layers,
            ctx_size=payload.ctx_size,
            hf_mmproj=payload.hf_mmproj,
            cpu_moe=payload.cpu_moe,
            flash_attn=payload.flash_attn,
            cache_type_k=payload.cache_type_k,
            cache_type_v=payload.cache_type_v,
            spec_type=payload.spec_type,
            spec_draft_n_max=payload.spec_draft_n_max,
            spec_draft_p_min=payload.spec_draft_p_min,
            spec_type_secondary=payload.spec_type_secondary,
            spec_ngram_mod_n_match=payload.spec_ngram_mod_n_match,
            jinja=payload.jinja,
            chat_template_kwargs=payload.chat_template_kwargs,
            reasoning_budget=payload.reasoning_budget,
            reasoning_budget_message=payload.reasoning_budget_message,
            temperature=payload.temperature,
            min_p=payload.min_p,
            top_k=payload.top_k,
            top_p=payload.top_p,
            presence_penalty=payload.presence_penalty,
            hf_repo=payload.hf_repo,
            hf_file=payload.hf_file,
            log_dir=str(Path(__file__).resolve().parent / "logs")
        )

        if server.is_port_in_use(): return {"status": "active", "message": f"Inference process already online at {payload.host}:{payload.port}."}
        if server.start(): return {"status": "success", "message": f"Engine {payload.model_type.upper()} spawned successfully on {payload.host}:{payload.port}."}
        else: raise HTTPException(status_code=500, detail="Hardware optimization binary timeout.")

    except HTTPException: raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Host Execution Crash: {str(e)}")

@app.post("/api/node/stop")
async def stop_node(port: int, host: str = "127.0.0.1"):
    server = LlamaServer(port=port, host=host)
    server.stop()
    log_file = Path(__file__).resolve().parent / "logs" / f"server_{port}.log"
    if log_file.exists():
        try: log_file.unlink()
        except: pass
    return {"status": "stopped", "message": f"Server halted cleanly."}

@app.get("/api/node/logs")
async def fetch_logs(port: int, max_lines: int = 6):
    log_file = Path(__file__).resolve().parent / "logs" / f"server_{port}.log"
    if not log_file.exists(): return {"logs": "waiting for startup"}
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
        if content.strip():
            lines = content.splitlines()
            filtered_lines = [l for l in lines if not any(x in l for x in ["GiB", "MiB", "%", "downloading", "download"])]
            output = filtered_lines[-max_lines:] if max_lines > 0 else filtered_lines
            return {"logs": "\n".join(output) if output else "waiting for startup"}
        return {"logs": "waiting for startup"}
    except Exception as e: return {"logs": f"[error parsing remote logs: {str(e)}]"}

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50000)
    args = parser.parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)