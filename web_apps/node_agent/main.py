import os
import sys
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent  
model_host_dir = project_root / "web_apps" / "utils" / "model_host"
if str(model_host_dir) not in sys.path:
    sys.path.insert(0, str(model_host_dir))

from lm_server_helper import LlamaServer

app = FastAPI(title="Infrastructure Compute Node Agent Daemon")

class StartNodePayload(BaseModel):
    model_type: str
    port: int                     # Dynamic Port
    host: str = "127.0.0.1"       # Dynamic IP (e.g. 100.96.40.81)
    hf_repo: str
    hf_file: str
    n_gpu_layers: int = -1
    ctx_size: int = 2048
    temperature: float = 0.1
    min_p: float = 0.05           # New Parameter
    thinking: bool = False

@app.get("/api/node/status")
async def check_status(port: int, host: str = "127.0.0.1"):
    server = LlamaServer(port=port, host=host)
    return {"online": server.is_port_in_use()}

@app.post("/api/node/start")
async def start_node(payload: StartNodePayload):
    # 1. Prevent the 500 error if UI fields are blank
    if not payload.hf_repo or not payload.hf_file:
        raise HTTPException(status_code=400, detail="Missing Hugging Face repo or file name. Please fill these out in the Client Settings.")

    # 2. Pass the dynamic UI parameters directly to the server helper
    server = LlamaServer(
        host=payload.host,      
        port=payload.port,
        n_gpu_layers=payload.n_gpu_layers,
        ctx_size=payload.ctx_size,
        thinking=payload.thinking,
        temperature=payload.temperature,
        min_p=payload.min_p,
        hf_repo=payload.hf_repo,
        hf_file=payload.hf_file,
        log_dir=str(Path(__file__).resolve().parent / "logs")
    )

    if server.is_port_in_use():
        return {"status": "active", "message": f"Inference process already online at {payload.host}:{payload.port}."}

    # 3. Safely catch any internal binary errors
    try:
        if server.start():
            return {"status": "success", "message": f"Engine {payload.model_type.upper()} spawned successfully on {payload.host}:{payload.port}."}
        else:
            raise HTTPException(status_code=500, detail="Hardware optimization binary timeout.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/node/stop")
async def stop_node(port: int, host: str = "127.0.0.1"):
    server = LlamaServer(port=port, host=host)
    server.stop()
    log_file = Path(__file__).resolve().parent / "logs" / f"server_{port}.log"
    if log_file.exists():
        try: log_file.unlink()
        except: pass
    return {"status": "stopped", "message": f"Server process on port {port} halted cleanly."}

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
    except Exception as e:
        return {"logs": f"[error parsing remote logs: {str(e)}]"}

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50000)
    args = parser.parse_args()
    
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)