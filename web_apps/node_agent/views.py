import os
import sys
import socket
import subprocess
import signal
from pathlib import Path
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse

project_root = Path(__file__).resolve().parent.parent.parent

if not hasattr(sys, '_active_host_daemons'):
    sys._active_host_daemons = {}

running_processes = sys._active_host_daemons


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def host_dashboard(request):
    current_ip = get_local_ip()

    if request.method == "POST":
        service_type = request.POST.get("service_type", "llm_agent").strip()
        host = request.POST.get("host", "0.0.0.0").strip()
        port = int(request.POST.get("port", 8705))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_port_blocked = s.connect_ex(("127.0.0.1", port)) == 0

        if is_port_blocked or port in running_processes:
            messages.error(request, f"Port {port} is currently occupied.")
            return redirect("host_dashboard")

        if service_type == "sam3_service":
            script_path = project_root / "web_apps" / "utils" / "model_host" / "sam3_server.py"
            display_name = "SAM3"
        else:
            script_path = project_root / "web_apps" / "node_agent" / "main.py"
            display_name = "Language Model"

        proc = subprocess.Popen(
            [sys.executable, str(script_path), "--host", host, "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, cwd=str(script_path.parent)
        )

        running_processes[port] = {
            "proc": proc,
            "type": display_name,
            "host": host
        }

        messages.success(request, f"Started {display_name} on {host}:{port}")
        return redirect("host_dashboard")

    active_list = []
    dead_ports = []
    for port, info in running_processes.items():
        if info["proc"].poll() is not None:
            dead_ports.append(port)
        else:
            active_list.append({
                "port": port, "type": info["type"], "host": info["host"]
            })
    for p in dead_ports:
        del running_processes[p]

    return render(request, "node_agent/dashboard.html", {
        "current_ip": current_ip,
        "active_daemons": active_list
    })


def stop_daemon_ui(request, port):
    if port in running_processes:
        proc = running_processes[port]["proc"]
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except:
            proc.kill()
        del running_processes[port]
        messages.success(request, f"Closed listener on port {port}")
    else:
        try:
            if sys.platform != "win32":
                out = subprocess.check_output(["lsof", "-t", f"-i:{port}"], text=True).strip()
                for pid in out.split("\n"):
                    if pid.strip(): os.kill(int(pid.strip()), signal.SIGKILL)
            messages.success(request, f"Cleared socket connection on port {port}")
        except:
            messages.error(request, f"No process found on port {port}")
            
    return redirect("host_dashboard")

def api_node_status(request):
    port = int(request.GET.get("port", 8080))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return JsonResponse({"online": s.connect_ex(("127.0.0.1", port)) == 0})

def api_node_start(request):
    return JsonResponse({"status": "success", "message": "Managed by host configuration manager."})

def api_node_stop(request):
    port = int(request.GET.get("port", 8080))
    return stop_daemon_ui(request, port)

def api_node_logs(request):
    return JsonResponse({"logs": "Managed by host agent log subsystem."})