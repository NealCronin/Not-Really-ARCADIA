import json
import os
import socket
import sys
import threading
import requests
import subprocess
from pathlib import Path
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse

project_root = Path(__file__).resolve().parent.parent.parent  
model_host_dir = project_root / "web_apps" / "utils" / "model_host"

if str(model_host_dir) not in sys.path:
    sys.path.insert(0, str(model_host_dir))

from lm_server_helper import LlamaServer

active_vpn_tunnels = {}


class VPNTunnelBridge:
    def __init__(self, local_host: str, local_port: int, remote_host: str, remote_port: int):
        self.local_host = local_host
        self.local_port = local_port
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.running = False
        self.server_socket = None

    def _pipe(self, source, destination):
        try:
            while self.running:
                data = source.recv(262144)
                if not data: break
                destination.sendall(data)
        except: pass
        finally:
            try: source.close()
            except: pass
            try: destination.close()
            except: pass

    def _handle_client(self, client_socket, addr):
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            remote_socket.connect((self.remote_host, self.remote_port))
        except:
            client_socket.close()
            return
        threading.Thread(target=self._pipe, args=(client_socket, remote_socket), daemon=True).start()
        threading.Thread(target=self._pipe, args=(remote_socket, client_socket), daemon=True).start()

    def start(self) -> bool:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.local_host, self.local_port))
            self.server_socket.listen(100)
            self.running = True
        except:
            return False
        def listen_loop():
            while self.running:
                try:
                    client_conn, addr = self.server_socket.accept()
                    self._handle_client(client_conn, addr)
                except: break
        threading.Thread(target=listen_loop, name=f"Bridge-{self.local_port}", daemon=True).start()
        return True

    def stop(self):
        self.running = False
        if self.server_socket:
            try: self.server_socket.close()
            except: pass


def check_server_alive(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex((host, port)) == 0
    except: return False


def local_directory_browser_api(request):
    try:
        script = 'tell application "System Events" to activate\nPOSIX path of (choose folder with prompt "Select Dataset Root Location:")'
        proc = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        if proc.returncode == 0 and stdout.strip():
            selected_path = stdout.strip()
            return JsonResponse({"success": True, "path": selected_path, "basename": os.path.basename(selected_path) or selected_path})
        return JsonResponse({"success": False, "error": "Canceled by user"})
    except Exception as e: return JsonResponse({"success": False, "error": str(e)})


def config_page(request):
    lm_json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    
    if lm_json_path.exists():
        with open(lm_json_path, "r") as f: lm_config = json.load(f)
    else:
        lm_config = {
            "ENVIRONMENT_SETTINGS": {"DATASET_ROOT": "", "NODE_AGENT_PORT": 8705, "recent_dataset_paths": []},
            "VLM_CONFIG": {}, "LLM_CONFIG": {}, "CONV_LLM_CONFIG": {}, "SAM3_CONFIG": {}
        }

    env_settings = lm_config.setdefault("ENVIRONMENT_SETTINGS", {"DATASET_ROOT": "", "NODE_AGENT_PORT": 8705, "recent_dataset_paths": []})

    if request.method == "POST":
        chosen_dataset_root = request.POST.get("dataset_root", "").strip()
        node_agent_port = int(request.POST.get("node_agent_port", 8705))
        
        env_settings["DATASET_ROOT"] = chosen_dataset_root
        env_settings["NODE_AGENT_PORT"] = node_agent_port
        
        if chosen_dataset_root and chosen_dataset_root not in env_settings.get("recent_dataset_paths", []):
            if "recent_dataset_paths" not in env_settings: env_settings["recent_dataset_paths"] = []
            env_settings["recent_dataset_paths"].insert(0, chosen_dataset_root)
            env_settings["recent_dataset_paths"] = env_settings["recent_dataset_paths"][:10]

        for block in ["VLM_CONFIG", "LLM_CONFIG", "CONV_LLM_CONFIG", "SAM3_CONFIG"]:
            pfx = block.lower().replace("_config", "")
            if block == "SAM3_CONFIG":
                lm_config[block] = {
                    "mode": request.POST.get("sam3_mode", "local"),
                    "host": request.POST.get("sam3_host", "127.0.0.1").strip(),
                    "port": int(request.POST.get("sam3_port") or 8000),
                    "use_vpn_tunnel": request.POST.get("sam3_use_vpn") in ["true", "on"],
                    "vpn_host": request.POST.get("sam3_vpn_host", "").strip()
                }
            else:
                lm_config[block] = {
                    "mode": request.POST.get(f"{pfx}_mode", "local"),
                    "host": request.POST.get(f"{pfx}_host", "127.0.0.1").strip(),
                    "port": int(request.POST.get(f"{pfx}_port", 8080)),
                    "use_vpn_tunnel": request.POST.get(f"{pfx}_use_vpn") in ["true", "on"],
                    "vpn_host": request.POST.get(f"{pfx}_vpn_host", "").strip(),
                    "n_gpu_layers": int(request.POST.get(f"{pfx}_ngl", 0)),
                    "local_model_path": "", "local_mmproj_path": "",
                    "hf_repo": request.POST.get(f"{pfx}_hf_repo", "").strip(),
                    "hf_file": request.POST.get(f"{pfx}_hf_file", "").strip(),
                    "hf_mmproj_file": request.POST.get(f"{pfx}_hf_mmproj_file", "").strip() if pfx == "vlm" else "",
                    "max_tokens": int(request.POST.get(f"{pfx}_max_tokens", 2048)),
                    "ctx_size": int(request.POST.get(f"{pfx}_ctx_size", 4096)),
                    "temperature": float(request.POST.get(f"{pfx}_temp", 0.1 if pfx != "conv" else 0.7)),
                    "thinking": request.POST.get(f"{pfx}_thinking") == "true",
                    "top_p": float(request.POST.get(f"{pfx}_top_p", 0.95)),
                    "min_p": float(request.POST.get(f"{pfx}_min_p", 0.05)),
                    "repeat_penalty": float(request.POST.get(f"{pfx}_repeat_penalty", 1.0))
                }

        with open(lm_json_path, "w") as f: json.dump(lm_config, f, indent=2)
        messages.success(request, "Parameters saved.")
        return redirect("config_page")

    past_dataset_paths_history = [{"full_path": p, "display_name": os.path.basename(p) or p} for p in env_settings.get("recent_dataset_paths", [])]
    return render(request, "infrastructure_manager/config.html", {
        "config": lm_config,
        "env_settings": env_settings,
        "current_dataset_root": env_settings.get("DATASET_ROOT", ""),
        "current_dataset_basename": os.path.basename(env_settings.get("DATASET_ROOT", "")) or env_settings.get("DATASET_ROOT", ""),
        "recent_dataset_paths": past_dataset_paths_history
    })


def restore_defaults(request):
    lm_json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    defaults_json_path = Path(settings.DRONE_CODE_DIR) / "config" / "defaults.json"
    
    if not defaults_json_path.exists():
        messages.error(request, "Defaults template file (defaults.json) is missing.")
        return redirect("config_page")
        
    try:
        with open(defaults_json_path, "r") as f: default_config = json.load(f)
        with open(lm_json_path, "w") as f: json.dump(default_config, f, indent=2)
        messages.success(request, "Restored to default configurations.")
    except Exception as e:
        messages.error(request, f"Error writing configuration: {str(e)}")
        
    return redirect("config_page")


def manage_models(request):
    json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    with open(json_path, "r") as f: lm_config = json.load(f)
    
    env_settings = lm_config.get("ENVIRONMENT_SETTINGS", {"NODE_AGENT_PORT": 8705})
    node_port = env_settings.get("NODE_AGENT_PORT", 8705)
        
    def fetch_stream(config_node):
        host = config_node.get("host", "127.0.0.1")
        port = int(config_node.get("port", 8080))
        mode = config_node.get("mode", "local")
        use_vpn = config_node.get("use_vpn_tunnel", False)
        vpn_host = config_node.get("vpn_host", "").strip()
        
        if mode == "remote":
            target_ip = vpn_host if use_vpn else host
            try:
                agent_res = requests.get(f"http://{target_ip}:{node_port}/api/node/logs?port={port}&max_lines=6", timeout=0.5).json()
                is_running_on_host = requests.get(f"http://{target_ip}:{node_port}/api/node/status?port={port}", timeout=0.5).json().get("online", False)
                if use_vpn:
                    if port in active_vpn_tunnels:
                        return True, f"[vpn bridge proxy operational]\n" + agent_res.get("logs", "waiting for startup")
                    return False, "waiting for startup (local tunnel proxy bridge offline)"
                return is_running_on_host, agent_res.get("logs", "waiting for startup")
            except:
                return False, f"[node_agent unreachable at http://{target_ip}:{node_port}]"

        is_online = check_server_alive(host, port)
        log_file = Path(settings.DRONE_CODE_DIR) / "logs" / f"server_{port}.log"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
                if content.strip():
                    lines = content.splitlines()
                    filtered_lines = [l for l in lines if not any(x in l for x in ["GiB", "MiB", "%", "downloading", "download"])]
                    if filtered_lines: return is_online, "\n".join(filtered_lines[-6:])
            except: pass
        return is_online, "waiting for startup"

    vlm_online, vlm_logs = fetch_stream(lm_config["VLM_CONFIG"])
    llm_online, llm_logs = fetch_stream(lm_config["LLM_CONFIG"])
    conv_online, conv_logs = fetch_stream(lm_config["CONV_LLM_CONFIG"])

    return render(request, "infrastructure_manager/manage_models.html", {
        "config": lm_config, "vlm_online": vlm_online, "vlm_logs": vlm_logs,
        "llm_online": llm_online, "llm_logs": llm_logs, "conv_online": conv_online, "conv_logs": conv_logs,
    })


def control_model(request, model_type, action):
    json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    with open(json_path, "r") as f: lm_config = json.load(f)
    
    env_settings = lm_config.get("ENVIRONMENT_SETTINGS", {"NODE_AGENT_PORT": 8705})
    node_port = env_settings.get("NODE_AGENT_PORT", 8705)
    
    key_map = {"vlm": "VLM_CONFIG", "llm": "LLM_CONFIG", "conv": "CONV_LLM_CONFIG", "sam3": "SAM3_CONFIG"}
    config_block = lm_config.get(key_map.get(model_type))
    
    host = config_block.get("host", "127.0.0.1")
    port = int(config_block.get("port", 8080))
    mode = config_block.get("mode", "local")
    use_vpn = config_block.get("use_vpn_tunnel", False)
    vpn_host = config_block.get("vpn_host", "").strip()

    if mode == "remote":
        target_ip = vpn_host if use_vpn else host
        try:
            if action == "start":
                payload = {
                    "model_type": model_type, "hf_repo": config_block.get("hf_repo", ""), "hf_file": config_block.get("hf_file", ""),
                    "n_gpu_layers": int(config_block.get("n_gpu_layers", 0)), "ctx_size": int(config_block.get("ctx_size", 4096)),
                    "temperature": float(config_block.get("temperature", 0.1)), "min_p": float(config_block.get("min_p", 0.05)),
                    "thinking": config_block.get("thinking", False)
                }
                requests.post(f"http://{target_ip}:{node_port}/api/node/start", json=payload, timeout=2.0)
                if use_vpn and port not in active_vpn_tunnels:
                    bridge = VPNTunnelBridge(local_host=host, local_port=port, remote_host=vpn_host, remote_port=port)
                    if bridge.start(): active_vpn_tunnels[port] = bridge
                messages.success(request, f"Start instruction sent.")
            elif action == "stop":
                requests.post(f"http://{target_ip}:{node_port}/api/node/stop?port={port}", timeout=2.0)
                if port in active_vpn_tunnels:
                    active_vpn_tunnels[port].stop()
                    del active_vpn_tunnels[port]
                messages.success(request, f"Stop instruction sent.")
        except:
            messages.error(request, f"Connection failed to host at {target_ip}:{node_port}")
        return redirect("manage_models")

    server = LlamaServer(
        host=host, port=port, n_gpu_layers=int(config_block.get("n_gpu_layers", -1)), ctx_size=int(config_block.get("ctx_size", 2048)),
        thinking=config_block.get("thinking", False), temperature=config_block.get("temperature"),
        hf_repo=config_block.get("hf_repo"), hf_file=config_block.get("hf_file"), hf_mmproj_file=config_block.get("hf_mmproj_file"),
        log_dir=str(Path(settings.DRONE_CODE_DIR) / "logs")
    )
    
    if action == "start":
        if server.is_port_in_use(): return redirect("manage_models")
        log_file = Path(server.log_dir) / f"server_{server.port}.log"
        if log_file.exists():
            try: log_file.unlink()
            except: pass
        threading.Thread(target=server.start, kwargs={"timeout": 120}, daemon=True).start()
        messages.success(request, f"Starting local server.")
    elif action == "stop":
        server.stop()
        log_file = Path(server.log_dir) / f"server_{server.port}.log"
        if log_file.exists():
            try: log_file.unlink()
            except: pass
        messages.success(request, f"Stopped local server.")
    return redirect("manage_models")


def view_terminal(request, model_type):
    json_path = Path(settings.DRONE_CODE_DIR) / "config" / "lm_config.json"
    with open(json_path, "r") as f: lm_config = json.load(f)
    
    env_settings = lm_config.get("ENVIRONMENT_SETTINGS", {"NODE_AGENT_PORT": 8705})
    node_port = env_settings.get("NODE_AGENT_PORT", 8705)
    
    key_map = {"vlm": "VLM_CONFIG", "llm": "LLM_CONFIG", "conv": "CONV_LLM_CONFIG"}
    config_block = lm_config.get(key_map.get(model_type))
    host = config_block.get("host", "127.0.0.1")
    port = config_block.get("port", 8080)
    mode = config_block.get("mode", "local")
    use_vpn = config_block.get("use_vpn_tunnel", False)
    vpn_host = config_block.get("vpn_host", "").strip()
    
    if mode == "remote":
        target_ip = vpn_host if use_vpn else host
        try:
            res = requests.get(f"http://{target_ip}:{node_port}/api/node/logs?port={port}&max_lines=0", timeout=1.0).json()
            is_online = requests.get(f"http://{target_ip}:{node_port}/api/node/status?port={port}", timeout=1.0).json().get("online", False)
            log_content = res.get("logs", "waiting for startup")
        except:
            is_online = False
            log_content = f"[node_agent unreachable at http://{target_ip}:{node_port}]"
    else:
        is_online = check_server_alive(host, port)
        log_file = Path(settings.DRONE_CODE_DIR) / "logs" / f"server_{port}.log"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
                if content.strip():
                    lines = content.splitlines()
                    filtered_lines = [l for l in lines if not any(x in l for x in ["GiB", "MiB", "%", "downloading", "download"])]
                    log_content = "\n".join(filtered_lines) if filtered_lines else "waiting for startup"
                else: log_content = "waiting for startup"
            except Exception as e: log_content = f"[error reading trace: {e}]"
        else: log_content = "waiting for startup"
            
    return render(request, "infrastructure_manager/terminal.html", {
        "model_name": model_type.upper(), "port": port, "is_online": is_online, "log_content": log_content
    })