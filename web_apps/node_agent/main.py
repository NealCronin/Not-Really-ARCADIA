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
    port: int          # <-- Added dynamic port
    hf_repo: str
    hf_file: str
    n_gpu_layers: int = -1
    ctx_size: int = 2048
    temperature: float = 0.1
    min_p: float = 0.05
    thinking: bool = False


@app.get("/api/node/status")
async def check_status(port: int):
    server = LlamaServer(port=port)
    return {"online": server.is_port_in_use()}


@app.post("/api/node/start")
async def start_node(payload: StartNodePayload):
    # 1. The hardcoded port_map is completely gone.
    # We now strictly trust the port sent by the client.
    target_port = payload.port

    print("\n" + "="*50)
    print(f"📡 [NODE AGENT] RECEIVED START COMMAND")
    print(f"Target Engine: {payload.model_type.upper()} on PORT {target_port}")
    print(f"Payload Data: {payload.model_dump_json(indent=2)}")
    print("="*50 + "\n")

    server = LlamaServer(
        host="0.0.0.0", 
        port=target_port,  # <-- Injected dynamically here
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
        return {"status": "active", "message": f"Inference process already online at port {target_port}."}

    if server.start():
        return {"status": "success", "message": f"Engine {payload.model_type.upper()} spawned successfully on port {target_port}."}
    else:
        raise HTTPException(status_code=500, detail="Hardware optimization binary timeout.")


@app.post("/api/node/stop")
async def stop_node(port: int):
    server = LlamaServer(port=port)
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
    
    # Spin up our standard web server framework cleanly
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)