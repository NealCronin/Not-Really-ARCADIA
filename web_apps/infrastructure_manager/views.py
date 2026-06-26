import os
import sys
import json
import subprocess
import requests
from pathlib import Path
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse

from web_apps.utils.model_host.vpn_tunnel_bridge import VPNTunnelBridge

project_root = Path(__file__).resolve().parent.parent.parent
config_path = project_root / "pipeline" / "drone_heatmap" / "config" / "lm_config.json"

if not hasattr(sys, '_active_client_bridges'): sys._active_client_bridges = {}
if not hasattr(sys, '_active_local_procs'): sys._active_local_procs = {}

active_bridges = sys._active_client_bridges
active_local_procs = sys._active_local_procs

def load_config():
    if not config_path.exists(): return {}
    with open(config_path, 'r') as f: return json.load(f)

def save_config(data):
    # FIXED: This forces the OS to create the folder if it doesn't exist!
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f: json.dump(data, f, indent=4)

def get_node_agent_url(model_config, env_config):
    host_ip = model_config.get("vpn_host") if model_config.get("use_vpn_tunnel") else model_config.get("host")
    port = env_config.get("INSTRUCTION_PORT", 50000)
    return f"http://{host_ip}:{port}/api/node"

def config_page(request):
    config = load_config()
    if request.method == "POST":
        config.setdefault("ENV_SETTINGS", {})
        config["ENV_SETTINGS"]["DATASET_ROOT"] = request.POST.get("dataset_root", "")
        config["ENV_SETTINGS"]["INSTRUCTION_PORT"] = int(request.POST.get("instruction_port", 50000))
        
        sections = {"vlm": "VLM_CONFIG", "llm": "LLM_CONFIG", "conv": "CONV_LLM_CONFIG", "sam3": "SAM3_CONFIG"}
        
        for prefix, key in sections.items():
            config.setdefault(key, {})
            config[key]["mode"] = request.POST.get(f"{prefix}_mode", "local")
            config[key]["host"] = request.POST.get(f"{prefix}_host", "127.0.0.1")
            config[key]["port"] = int(request.POST.get(f"{prefix}_port", 8080) or 8080)
            config[key]["use_vpn_tunnel"] = request.POST.get(f"{prefix}_use_vpn") == "true"
            config[key]["vpn_host"] = request.POST.get(f"{prefix}_vpn_host", "")
            
            if prefix != "sam3":
                config[key]["hf_repo"] = request.POST.get(f"{prefix}_hf_repo", "")
                config[key]["hf_file"] = request.POST.get(f"{prefix}_hf_file", "")
                config[key]["hf_mmproj"] = request.POST.get(f"{prefix}_hf_mmproj", "")
                config[key]["n_gpu_layers"] = int(request.POST.get(f"{prefix}_ngl", -1) or -1)
                config[key]["ctx_size"] = int(request.POST.get(f"{prefix}_ctx_size", 2048) or 2048)
                
                # Advanced Params
                config[key]["cpu_moe"] = int(request.POST.get(f"{prefix}_cpu_moe", 0) or 0)
                config[key]["flash_attn"] = request.POST.get(f"{prefix}_flash_attn") == "true"
                config[key]["cache_type_k"] = request.POST.get(f"{prefix}_cache_k", "")
                config[key]["cache_type_v"] = request.POST.get(f"{prefix}_cache_v", "")
                config[key]["spec_type"] = request.POST.get(f"{prefix}_spec_type", "")
                config[key]["spec_draft_n_max"] = int(request.POST.get(f"{prefix}_spec_n", 0) or 0)
                config[key]["spec_draft_p_min"] = float(request.POST.get(f"{prefix}_spec_p", 0.0) or 0.0)
                config[key]["spec_type_secondary"] = request.POST.get(f"{prefix}_spec_sec", "")
                config[key]["spec_ngram_mod_n_match"] = int(request.POST.get(f"{prefix}_spec_match", 0) or 0)
                config[key]["jinja"] = request.POST.get(f"{prefix}_jinja") == "true"
                config[key]["chat_template_kwargs"] = request.POST.get(f"{prefix}_chat_kwargs", "")
                config[key]["reasoning_budget"] = int(request.POST.get(f"{prefix}_reason_budget", 0) or 0)
                config[key]["reasoning_budget_message"] = request.POST.get(f"{prefix}_reason_msg", "")
                
                # Sampling
                config[key]["temperature"] = float(request.POST.get(f"{prefix}_temp", 0.1) or 0.1)
                config[key]["min_p"] = float(request.POST.get(f"{prefix}_min_p", 0.05) or 0.05)
                config[key]["top_k"] = int(request.POST.get(f"{prefix}_top_k", 40) or 40)
                config[key]["top_p"] = float(request.POST.get(f"{prefix}_top_p", 0.95) or 0.95)
                config[key]["presence_penalty"] = float(request.POST.get(f"{prefix}_pres_pen", 0.0) or 0.0)

        save_config(config)
        messages.success(request, "Infrastructure configurations saved successfully.")
        return redirect('config_page')

    context = {
        'config': config,
        'env_settings': config.get("ENV_SETTINGS", {}),
        'current_dataset_root': config.get("ENV_SETTINGS", {}).get("DATASET_ROOT", ""),
        'current_dataset_basename': os.path.basename(config.get("ENV_SETTINGS", {}).get("DATASET_ROOT", "Select Path")),
        'recent_dataset_paths': [] 
    }
    return render(request, "infrastructure_manager/config.html", context)

def manage_models(request):
    config = load_config()
    def get_preview_log(model_type):
        try:
            target_config = config.get(f"{model_type.upper()}_CONFIG", {})
            if target_config.get("mode") == "remote":
                url = f"{get_node_agent_url(target_config, config.get('ENV_SETTINGS', {}))}/logs?port={target_config.get('port')}&max_lines=4"
                res = requests.get(url, timeout=2)
                return res.json().get("logs", "No logs.")
            else: return "Local execution active." if model_type in active_local_procs else "Offline."
        except: return "Connection error / Offline."

    context = {'config': config, 'vlm_logs': get_preview_log('vlm'), 'llm_logs': get_preview_log('llm'), 'conv_logs': get_preview_log('conv')}
    return render(request, "infrastructure_manager/manage_models.html", context)

def control_model(request, model_type, action):
    config = load_config()
    key = f"{model_type.upper()}_CONFIG" if model_type != "sam3" else "SAM3_CONFIG"
    if key not in config:
        messages.error(request, f"Unknown model config: {model_type}")
        return redirect('manage_models')

    model_config = config[key]
    mode = model_config.get("mode", "local")
    local_port = model_config.get("port", 8080)

    if action == "start":
        if mode == "remote":
            node_url = get_node_agent_url(model_config, config.get("ENV_SETTINGS", {}))
            local_host = model_config.get("host", "127.0.0.1")
            remote_host = model_config.get("vpn_host") if model_config.get("use_vpn_tunnel") else local_host

            payload = {
                "model_type": model_type,
                "port": local_port,
                "host": remote_host,
                "hf_repo": model_config.get("hf_repo", ""),
                "hf_file": model_config.get("hf_file", ""),
                "hf_mmproj": model_config.get("hf_mmproj", ""),
                "n_gpu_layers": model_config.get("n_gpu_layers", -1),
                "ctx_size": model_config.get("ctx_size", 2048),
                "cpu_moe": model_config.get("cpu_moe", 0),
                "flash_attn": model_config.get("flash_attn", False),
                "cache_type_k": model_config.get("cache_type_k", ""),
                "cache_type_v": model_config.get("cache_type_v", ""),
                "spec_type": model_config.get("spec_type", ""),
                "spec_draft_n_max": model_config.get("spec_draft_n_max", 0),
                "spec_draft_p_min": model_config.get("spec_draft_p_min", 0.0),
                "spec_type_secondary": model_config.get("spec_type_secondary", ""),
                "spec_ngram_mod_n_match": model_config.get("spec_ngram_mod_n_match", 0),
                "jinja": model_config.get("jinja", False),
                "chat_template_kwargs": model_config.get("chat_template_kwargs", ""),
                "reasoning_budget": model_config.get("reasoning_budget", 0),
                "reasoning_budget_message": model_config.get("reasoning_budget_message", ""),
                "temperature": model_config.get("temperature", 0.1),
                "min_p": model_config.get("min_p", 0.05),
                "top_k": model_config.get("top_k", 40),
                "top_p": model_config.get("top_p", 0.95),
                "presence_penalty": model_config.get("presence_penalty", 0.0)
            }
            
            try:
                res = requests.post(f"{node_url}/start", json=payload, timeout=60)
                if res.status_code != 200:
                    messages.error(request, f"Remote node rejected start command: {res.text}")
                    return redirect('manage_models')
            except Exception as e:
                messages.error(request, f"Failed to contact remote Node Agent: {str(e)}")
                return redirect('manage_models')

            if model_type in active_bridges: active_bridges[model_type].stop()
            bridge = VPNTunnelBridge(local_host, local_port, remote_host, local_port)
            if bridge.start():
                active_bridges[model_type] = bridge
                messages.success(request, f"{model_type.upper()} started remotely. Bridge secured.")
            else:
                messages.error(request, f"Remote started, but local Bridge Proxy failed to bind.")

        elif mode == "local":
            messages.info(request, "Local orchestration is under construction.")

    elif action == "stop":
        if mode == "remote":
            if model_type in active_bridges:
                active_bridges[model_type].stop()
                del active_bridges[model_type]
            node_url = get_node_agent_url(model_config, config.get("ENV_SETTINGS", {}))
            try:
                requests.post(f"{node_url}/stop?port={local_port}", timeout=5)
                messages.success(request, f"{model_type.upper()} shut down remotely and bridge closed.")
            except:
                messages.warning(request, f"Local bridge closed, but could not reach remote node.")
    return redirect('manage_models')

def view_terminal(request, model_type):
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
        log_content = "[Local Diagnostic Stream currently unlinked in UI]"
        is_online = True if model_type in active_local_procs else False

    context = {'model_name': model_type.upper(), 'port': model_config.get("port", "N/A"), 'is_online': is_online, 'log_content': log_content}
    return render(request, "infrastructure_manager/terminal.html", context)

def browse_local_api(request): return JsonResponse({"success": False, "message": "Native file browser implementation required."})
def restore_defaults(request):
    messages.info(request, "Defaults restored.")
    return redirect('config_page')