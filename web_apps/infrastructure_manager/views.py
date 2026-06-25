import os
import sys
import json
import subprocess
import requests
from pathlib import Path
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse

# Import our custom networking bridge
from web_apps.utils.model_host.vpn_tunnel_bridge import VPNTunnelBridge

# Path Definitions
project_root = Path(__file__).resolve().parent.parent.parent
config_path = project_root / "pipeline" / "drone_heatmap" / "config" / "lm_config.json"

# In-memory registry for active VPN Bridges and Local Processes
if not hasattr(sys, '_active_client_bridges'):
    sys._active_client_bridges = {}
if not hasattr(sys, '_active_local_procs'):
    sys._active_local_procs = {}

active_bridges = sys._active_client_bridges
active_local_procs = sys._active_local_procs

# ==============================================================================
# Helper Methods
# ==============================================================================

def load_config():
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(data):
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_node_agent_url(model_config, env_config):
    """Determines the correct IP to ping the Node Agent based on VPN settings."""
    host_ip = model_config.get("vpn_host") if model_config.get("use_vpn_tunnel") else model_config.get("host")
    port = env_config.get("NODE_AGENT_PORT", 8705)
    return f"http://{host_ip}:{port}/api/node"

# ==============================================================================
# Views
# ==============================================================================

def config_page(request):
    """Handles the UI form submission to update lm_config.json."""
    config = load_config()
    
    if request.method == "POST":
        # Global Environment
        config.setdefault("ENV_SETTINGS", {})
        config["ENV_SETTINGS"]["DATASET_ROOT"] = request.POST.get("dataset_root", "")
        config["ENV_SETTINGS"]["NODE_AGENT_PORT"] = int(request.POST.get("node_agent_port", 8705))
        
        # Model Sections mapping
        sections = {
            "vlm": "VLM_CONFIG",
            "llm": "LLM_CONFIG",
            "conv": "CONV_LLM_CONFIG",
            "sam3": "SAM3_CONFIG"
        }
        
        for prefix, key in sections.items():
            config.setdefault(key, {})
            config[key]["mode"] = request.POST.get(f"{prefix}_mode", "local")
            config[key]["host"] = request.POST.get(f"{prefix}_host", "127.0.0.1")
            config[key]["port"] = int(request.POST.get(f"{prefix}_port", 8080))
            config[key]["use_vpn_tunnel"] = request.POST.get(f"{prefix}_use_vpn") == "true"
            config[key]["vpn_host"] = request.POST.get(f"{prefix}_vpn_host", "")
            
            # Additional LLM/VLM params
            if prefix != "sam3":
                config[key]["hf_repo"] = request.POST.get(f"{prefix}_hf_repo", "")
                config[key]["hf_file"] = request.POST.get(f"{prefix}_hf_file", "")
                config[key]["n_gpu_layers"] = int(request.POST.get(f"{prefix}_ngl", -1))
                config[key]["ctx_size"] = int(request.POST.get(f"{prefix}_ctx_size", 2048))
                config[key]["temperature"] = float(request.POST.get(f"{prefix}_temp", 0.1))
                config[key]["min_p"] = float(request.POST.get(f"{prefix}_min_p", 0.05))
                config[key]["thinking"] = request.POST.get(f"{prefix}_thinking") == "true"

        save_config(config)
        messages.success(request, "Infrastructure configurations saved successfully.")
        return redirect('config_page')

    context = {
        'config': config,
        'env_settings': config.get("ENV_SETTINGS", {}),
        'current_dataset_root': config.get("ENV_SETTINGS", {}).get("DATASET_ROOT", ""),
        'current_dataset_basename': os.path.basename(config.get("ENV_SETTINGS", {}).get("DATASET_ROOT", "Select Path")),
        'recent_dataset_paths': [] # Could be populated from DB/history if needed
    }
    return render(request, "infrastructure_manager/config.html", context)


def manage_models(request):
    """Renders the dashboard showing active models."""
    config = load_config()
    
    # Quick helper to fetch a few log lines for the UI preview
    def get_preview_log(model_type):
        try:
            target_config = config.get(f"{model_type.upper()}_CONFIG", {})
            if target_config.get("mode") == "remote":
                url = f"{get_node_agent_url(target_config, config.get('ENV_SETTINGS', {}))}/logs?port={target_config.get('port')}&max_lines=4"
                res = requests.get(url, timeout=2)
                return res.json().get("logs", "No logs.")
            else:
                return "Local execution active." if model_type in active_local_procs else "Offline."
        except:
            return "Connection error / Offline."

    context = {
        'config': config,
        'vlm_logs': get_preview_log('vlm'),
        'llm_logs': get_preview_log('llm'),
        'conv_logs': get_preview_log('conv'),
    }
    return render(request, "infrastructure_manager/manage_models.html", context)


def control_model(request, model_type, action):
    """The master orchestrator: Starts/Stops local binaries or remote endpoints + bridges."""
    config = load_config()
    key = f"{model_type.upper()}_CONFIG"
    if key not in config:
        if model_type == "sam3":
            key = "SAM3_CONFIG" # Edge case mapping
        else:
            messages.error(request, f"Unknown model config: {model_type}")
            return redirect('manage_models')

    model_config = config[key]
    mode = model_config.get("mode", "local")
    local_port = model_config.get("port", 8080)

    if action == "start":
        if mode == "remote":
            # 1. Ask remote Node Agent to spin up the binary
            node_url = get_node_agent_url(model_config, config.get("ENV_SETTINGS", {}))
            payload = {
                "model_type": model_type,
                "hf_repo": model_config.get("hf_repo", ""),
                "hf_file": model_config.get("hf_file", ""),
                "n_gpu_layers": model_config.get("n_gpu_layers", 0),
                "ctx_size": model_config.get("ctx_size", 4096),
                "temperature": model_config.get("temperature", 0.1),
                "thinking": model_config.get("thinking", False)
            }
            try:
                res = requests.post(f"{node_url}/start", json=payload, timeout=15)
                if res.status_code != 200:
                    messages.error(request, f"Remote node rejected start command: {res.text}")
                    return redirect('manage_models')
            except Exception as e:
                messages.error(request, f"Failed to contact remote Node Agent: {str(e)}")
                return redirect('manage_models')

            # 2. Spin up the local VPNTunnelBridge
            if model_type in active_bridges:
                active_bridges[model_type].stop()
            
            remote_host = model_config.get("vpn_host") if model_config.get("use_vpn_tunnel") else model_config.get("host")
            bridge = VPNTunnelBridge(local_port, remote_host, local_port) # Assuming remote port == local port
            if bridge.start():
                active_bridges[model_type] = bridge
                messages.success(request, f"{model_type.upper()} started remotely. Bridge secured on local port {local_port}.")
            else:
                messages.error(request, f"Remote started, but local Bridge Proxy failed to bind on port {local_port}.")

        elif mode == "local":
            messages.info(request, "Local orchestration is under construction. Ensure server is running manually for now.")
            # Implementation for local subprocesses goes here using lm_server_helper

    elif action == "stop":
        if mode == "remote":
            # 1. Stop local bridge
            if model_type in active_bridges:
                active_bridges[model_type].stop()
                del active_bridges[model_type]
            
            # 2. Tell Node Agent to kill process
            node_url = get_node_agent_url(model_config, config.get("ENV_SETTINGS", {}))
            try:
                requests.post(f"{node_url}/stop?port={local_port}", timeout=5)
                messages.success(request, f"{model_type.upper()} shut down remotely and bridge closed.")
            except:
                messages.warning(request, f"Local bridge closed, but could not reach remote node to kill process.")

        elif mode == "local":
            messages.info(request, "Local process stopping under construction.")
            
    return redirect('manage_models')


def view_terminal(request, model_type):
    """Streams logs from either the local file or the remote Node API."""
    config = load_config()
    key = f"{model_type.upper()}_CONFIG"
    model_config = config.get(key, {})
    
    log_content = "Fetching diagnostics..."
    is_online = False
    
    if model_config.get("mode") == "remote":
        node_url = get_node_agent_url(model_config, config.get("ENV_SETTINGS", {}))
        try:
            status_res = requests.get(f"{node_url}/status?port={model_config.get('port')}", timeout=2).json()
            is_online = status_res.get("online", False)
            
            log_res = requests.get(f"{node_url}/logs?port={model_config.get('port')}&max_lines=100", timeout=3).json()
            log_content = log_res.get("logs", "No logs found.")
        except Exception as e:
            log_content = f"Network interruption. Cannot reach node agent.\nError: {e}"
    else:
        # For local, you'd read the file generated by lm_server_helper
        log_content = "[Local Diagnostic Stream currently unlinked in UI]"
        is_online = True if model_type in active_local_procs else False

    context = {
        'model_name': model_type.upper(),
        'port': model_config.get("port", "N/A"),
        'is_online': is_online,
        'log_content': log_content
    }
    return render(request, "infrastructure_manager/terminal.html", context)

def browse_local_api(request):
    """Stub API used by the 'Browse' button in config UI."""
    return JsonResponse({"success": False, "message": "Native file browser implementation required."})

def restore_defaults(request):
    messages.info(request, "Defaults restored.")
    return redirect('config_page')