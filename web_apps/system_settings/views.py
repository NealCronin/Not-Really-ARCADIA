import json
import os
import socket
import sys
import threading
import subprocess
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages

# Dynamically resolve and inject the pipeline/llm_host module path relative to this file's position
system_settings_dir = Path(__file__).resolve().parent
web_apps_dir = system_settings_dir.parent
project_root = web_apps_dir.parent
llm_host_dir = project_root / "pipeline" / "llm_host"

if str(llm_host_dir) not in sys.path:
    sys.path.insert(0, str(llm_host_dir))

from llama_server_helper import LlamaServer


def check_server_alive(host: str, port: int) -> bool:
    """Performs a lightweight socket probe to check if a model server is actively listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex((host, port)) == 0
    except:
        return False


def local_directory_browser_api(request):
    """Spawns a native macOS Finder folder selection window and returns the selected absolute path."""
    try:
        # Calls the macOS AppleScript engine to raise a native system file browser window over the UI
        script = 'tell application "System Events" to activate\nPOSIX path of (choose folder with prompt "Select Dataset Root Location:")'
        proc = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        
        if proc.returncode == 0 and stdout.strip():
            selected_path = stdout.strip()
            return JsonResponse({
                "success": True, 
                "path": selected_path,
                "basename": os.path.basename(selected_path) or selected_path
            })
        else:
            # Handles cases where the user clicks 'Cancel' or closes the Finder window
            return JsonResponse({"success": False, "error": "Canceled by user"})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


def config_page(request):
    """Manages the systems setting dashboard by strictly isolating environment variables from model parameters."""
    lm_json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    gen_json_path = Path(settings.DRONE_CODE_DIR) / "config" / "settings.json"
    
    if gen_json_path.exists():
        with open(gen_json_path, "r") as f: gen_config = json.load(f)
    else:
        gen_config = {"DATASET_ROOT": "", "recent_dataset_paths": []}
        
    if lm_json_path.exists():
        with open(lm_json_path, "r") as f: lm_config = json.load(f)
    else:
        lm_config = {}

    if request.method == "POST":
        chosen_dataset_root = request.POST.get("dataset_root", "").strip()
        gen_config["DATASET_ROOT"] = chosen_dataset_root
        
        if "recent_dataset_paths" not in gen_config:
            gen_config["recent_dataset_paths"] = []
        if chosen_dataset_root and chosen_dataset_root not in gen_config["recent_dataset_paths"]:
            gen_config["recent_dataset_paths"].insert(0, chosen_dataset_root)
            gen_config["recent_dataset_paths"] = gen_config["recent_dataset_paths"][:10]

        lm_config["SAM3_CONFIG"] = {
            "mode": request.POST.get("sam3_mode", "local"),
            "host": request.POST.get("sam3_host", "127.0.0.1").strip(),
            "port": int(request.POST.get("sam3_port") or 8000)
        }
        
        lm_config["VLM_CONFIG"] = {
            "mode": request.POST.get("vlm_mode", "local"),
            "host": request.POST.get("vlm_host", "127.0.0.1").strip(),
            "port": int(request.POST.get("vlm_port", 8080)),
            "n_gpu_layers": int(request.POST.get("vlm_ngl", 0)),
            "local_model_path": "", "local_mmproj_path": "",
            "hf_repo": request.POST.get("vlm_hf_repo", "").strip(),
            "hf_file": request.POST.get("vlm_hf_file", "").strip(),
            "hf_mmproj_file": request.POST.get("vlm_hf_mmproj_file", "").strip(),
            "max_tokens": int(request.POST.get("vlm_max_tokens", 2048)),
            "ctx_size": int(request.POST.get("vlm_ctx_size", 4096)),
            "temperature": float(request.POST.get("vlm_temp", 0.1)),
            "thinking": request.POST.get("vlm_thinking") == "true",
            "top_p": float(request.POST.get("vlm_top_p", 0.95)),
            "min_p": float(request.POST.get("vlm_min_p", 0.05)),
            "repeat_penalty": float(request.POST.get("vlm_repeat_penalty", 1.0))
        }
        
        lm_config["LLM_CONFIG"] = {
            "mode": request.POST.get("llm_mode", "local"),
            "host": request.POST.get("llm_host", "127.0.0.1").strip(),
            "port": int(request.POST.get("llm_port", 8081)),
            "n_gpu_layers": int(request.POST.get("llm_ngl", 0)),
            "local_model_path": "",
            "hf_repo": request.POST.get("llm_hf_repo", "").strip(),
            "hf_file": request.POST.get("llm_hf_file", "").strip(),
            "max_tokens": int(request.POST.get("llm_max_tokens", 2048)),
            "ctx_size": int(request.POST.get("llm_ctx_size", 4096)),
            "temperature": float(request.POST.get("llm_temp", 0.1)),
            "thinking": request.POST.get("llm_thinking") == "true",
            "top_p": float(request.POST.get("llm_top_p", 0.95)),
            "min_p": float(request.POST.get("llm_min_p", 0.05)),
            "repeat_penalty": float(request.POST.get("llm_repeat_penalty", 1.0))
        }
        
        lm_config["CONV_LLM_CONFIG"] = {
            "mode": request.POST.get("conv_mode", "local"),
            "host": request.POST.get("conv_host", "127.0.0.1").strip(),
            "port": int(request.POST.get("conv_port", 8082)),
            "n_gpu_layers": int(request.POST.get("conv_ngl", 0)),
            "local_model_path": "",
            "hf_repo": request.POST.get("conv_hf_repo", "").strip(),
            "hf_file": request.POST.get("conv_hf_file", "").strip(),
            "max_tokens": int(request.POST.get("conv_max_tokens", 2048)),
            "ctx_size": int(request.POST.get("conv_ctx_size", 4096)),
            "temperature": float(request.POST.get("conv_temp", 0.7)),
            "thinking": request.POST.get("conv_thinking") == "true",
            "top_p": float(request.POST.get("conv_top_p", 0.95)),
            "min_p": float(request.POST.get("conv_min_p", 0.05)),
            "repeat_penalty": float(request.POST.get("conv_repeat_penalty", 1.0))
        }

        with open(gen_json_path, "w") as f: json.dump(gen_config, f, indent=2)
        with open(lm_json_path, "w") as f: json.dump(lm_config, f, indent=2)
            
        messages.success(request, "Environment settings and isolated model configurations updated cleanly!")
        return redirect("config_page")

    # Pack values into metadata objects separating absolute values from short folder basenames
    past_dataset_paths_history = []
    for p in gen_config.get("recent_dataset_paths", []):
        past_dataset_paths_history.append({
            "full_path": p,
            "display_name": os.path.basename(p) or p
        })

    context = {
        "config": lm_config,
        "current_dataset_root": gen_config.get("DATASET_ROOT", ""),
        "current_dataset_basename": os.path.basename(gen_config.get("DATASET_ROOT", "")) or gen_config.get("DATASET_ROOT", ""),
        "recent_dataset_paths": past_dataset_paths_history
    }
    return render(request, "system_settings/config.html", context)


def manage_models(request):
    """Monitors model gallery states by querying lm_config.json exclusively."""
    json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    with open(json_path, "r") as f:
        current_config = json.load(f)
        
    def fetch_real_terminal_stream(config_node):
        host = config_node.get("host", "127.0.0.1")
        port = int(config_node.get("port", 8080))
        mode = config_node.get("mode", "local")
        is_online = check_server_alive(host, port)
        
        if mode == "remote":
            if is_online:
                return True, f"[info] Proxy tracking remote engine link active at http://{host}:{port}"
            return False, "waiting for startup"
            
        log_file = Path(settings.DRONE_CODE_DIR) / "logs" / f"server_{port}.log"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    lines = content.splitlines()
                    filtered_lines = [
                        l for l in lines 
                        if not any(x in l for x in ["GiB", "MiB", "%", "downloading", "download"])
                    ]
                    if filtered_lines:
                        return is_online, "\n".join(filtered_lines[-6:])
            except:
                pass
        return is_online, "waiting for startup"

    vlm_online, vlm_logs = fetch_real_terminal_stream(current_config["VLM_CONFIG"])
    llm_online, llm_logs = fetch_real_terminal_stream(current_config["LLM_CONFIG"])
    conv_online, conv_logs = fetch_real_terminal_stream(current_config["CONV_LLM_CONFIG"])

    context = {
        "config": current_config,
        "vlm_online": vlm_online, "vlm_logs": vlm_logs,
        "llm_online": llm_online, "llm_logs": llm_logs,
        "conv_online": conv_online, "conv_logs": conv_logs,
    }
    return render(request, "system_settings/manage_models.html", context)

def control_model(request, model_type, action):
    json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    with open(json_path, "r") as f:
        current_config = json.load(f)
    key_map = {"vlm": "VLM_CONFIG", "llm": "LLM_CONFIG", "conv": "CONV_LLM_CONFIG"}
    config_block = current_config.get(key_map.get(model_type))
    
    server = LlamaServer(
        host=config_block.get("host", "127.0.0.1"), port=int(config_block.get("port", 8080)),
        n_gpu_layers=int(config_block.get("n_gpu_layers", -1)), ctx_size=int(config_block.get("ctx_size", 2048)),
        thinking=config_block.get("thinking", False), temperature=config_block.get("temperature"),
        top_p=config_block.get("top_p"), min_p=config_block.get("min_p"), repeat_penalty=config_block.get("repeat_penalty"),
        hf_repo=config_block.get("hf_repo"), hf_file=config_block.get("hf_file"), hf_mmproj_file=config_block.get("hf_mmproj_file"),
        log_dir=str(Path(settings.DRONE_CODE_DIR) / "logs")
    )
    
    if action == "start":
        if server.is_port_in_use():
            messages.success(request, f"Tracking core engine ({model_type.upper()}) is already active.")
            return redirect("manage_models")
            
        log_file = Path(server.log_dir) / f"server_{server.port}.log"
        if log_file.exists():
            try: log_file.unlink()
            except: pass

        bg_thread = threading.Thread(target=server.start, kwargs={"timeout": 120})
        bg_thread.daemon = True
        bg_thread.start()
        
        messages.success(request, f"Startup sequence initiated for {model_type.upper()}. Accessing Hugging Face repositories...")
        
    elif action == "stop":
        server.stop()
        log_file = Path(server.log_dir) / f"server_{server.port}.log"
        if log_file.exists():
            try: log_file.unlink()
            except: pass
        messages.success(request, f"Server operating on port {server.port} stopped cleanly.")
        
    return redirect("manage_models")

def view_terminal(request, model_type):
    json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    with open(json_path, "r") as f:
        current_config = json.load(f)
    key_map = {"vlm": "VLM_CONFIG", "llm": "LLM_CONFIG", "conv": "CONV_LLM_CONFIG"}
    config_block = current_config.get(key_map.get(model_type))
    host = config_block.get("host", "127.0.0.1")
    port = config_block.get("port", 8080)
    mode = config_block.get("mode", "local")
    
    is_online = check_server_alive(host, port)
    
    if mode == "remote":
        log_content = f"[info] Routing proxy to remote cluster at http://{host}:{port}" if is_online else "waiting for startup"
    else:
        log_file = Path(settings.DRONE_CODE_DIR) / "logs" / f"server_{port}.log"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    lines = content.splitlines()
                    filtered_lines = [
                        l for l in lines 
                        if not any(x in l for x in ["GiB", "MiB", "%", "downloading", "download"])
                    ]
                    log_content = "\n".join(filtered_lines) if filtered_lines else "waiting for startup"
                else:
                    log_content = "waiting for startup"
            except Exception as e:
                log_content = f"[error reading file system stream context: {e}]"
        else:
            log_content = "waiting for startup"
            
    return render(request, "system_settings/terminal.html", {
        "model_name": model_type.upper(), "port": port, "is_online": is_online, "log_content": log_content
    })