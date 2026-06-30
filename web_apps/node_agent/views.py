import os
import sys
import socket
import subprocess
import signal
from pathlib import Path
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse

# Import the new ActiveDaemon model
from .models import ActiveDaemon

project_root = Path(__file__).resolve().parent.parent.parent

def get_local_ip():
    """Fetches the actual local LAN IP address instead of relying on localhost."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Helper to check if a port is currently bound on the specified interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        check_host = "127.0.0.1" if host == "0.0.0.0" else host # <--- Add this safety check
        return s.connect_ex((check_host, port)) == 0

def is_pid_running(pid: int) -> bool:
    """Cross-platform check to see if a specific PID is still alive."""
    if not pid:
        return False
    try:
        if sys.platform == "win32":
            output = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True, text=True)
            return str(pid) in output
        else:
            os.kill(pid, 0) # Sending signal 0 checks existence without killing
        return True
    except OSError:
        return False

def host_dashboard(request):
    current_ip = get_local_ip()

    # 1. State Cleanup
    for daemon in ActiveDaemon.objects.all():
        if not is_pid_running(daemon.pid) and not is_port_in_use(daemon.port, daemon.host): # <--- Pass daemon.host
            daemon.delete()

    active_daemon = ActiveDaemon.objects.first()

    # 2. Handle Starting
    if request.method == "POST":
        if active_daemon:
            messages.error(request, "A dispatcher is already active. Close it first.")
            return redirect("host_dashboard")

        host = request.POST.get("host", "0.0.0.0").strip()
        port = int(request.POST.get("port", 50000))

        if is_port_in_use(port, host) or ActiveDaemon.objects.filter(port=port).exists(): # <--- Pass host
            messages.error(request, f"Port {port} is currently occupied on {host}.")
            return redirect("host_dashboard")

        # Always route to main.py
        script_path = project_root / "web_apps" / "node_agent" / "main.py"
        display_name = "Node Dispatcher"

        log_dir = project_root / "pipeline" / "drone_heatmap" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "node_agent.log", "a")

        proc = subprocess.Popen(
            [sys.executable, str(script_path), "--host", host, "--port", str(port)],
            stdout=log_file, 
            stderr=subprocess.STDOUT, 
            text=True, 
            cwd=str(script_path.parent)
        )

        ActiveDaemon.objects.create(
            service_type=display_name,
            host=host,
            port=port,
            pid=proc.pid
        )

        messages.success(request, f"Dispatcher bound to {host}:{port}")
        return redirect("host_dashboard")

    return render(request, "node_agent/dashboard.html", {
        "active_daemon": active_daemon
    })

def stop_daemon_ui(request, port):
    """Statelessly hunts down and terminates a process by its port or PID."""
    daemon = ActiveDaemon.objects.filter(port=port).first()
    killed_something = False
    
    # 1. Surgical Port Strike (OS Level)
    try:
        if sys.platform != "win32":
            out = subprocess.check_output(["lsof", "-t", f"-i:{port}"], text=True).strip()
            for pid_str in out.split("\n"):
                if pid_str.strip():
                    os.kill(int(pid_str.strip()), signal.SIGKILL)
                    killed_something = True
        else:
            out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True, text=True)
            for line in out.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    target_pid = parts[-1]
                    subprocess.run(["taskkill", "/F", "/PID", target_pid], check=True, stdout=subprocess.DEVNULL)
                    killed_something = True
    except Exception:
        pass # Port scan failed or came up empty

    # 2. Database Cleanup & PID Fallback
    if daemon:
        # If the port scan missed it, try killing the exact PID we saved
        if daemon.pid and not killed_something:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/PID", str(daemon.pid)], stdout=subprocess.DEVNULL)
                else:
                    os.kill(daemon.pid, signal.SIGKILL)
            except Exception:
                pass
        
        daemon.delete()
        messages.success(request, f"Closed listener on port {port}")
    elif killed_something:
        messages.success(request, f"Cleared socket connection on port {port} (Unregistered daemon)")
    else:
        messages.error(request, f"No active process found on port {port}")
        
    return redirect("host_dashboard")

# ==============================================================================
# Unified Network API Endpoints (Called remotely by Infrastructure Manager)
# ==============================================================================

def api_node_status(request):
    port = int(request.GET.get("port", 8080))
    return JsonResponse({"online": is_port_in_use(port)})

def api_node_start(request):
    return JsonResponse({"status": "success", "message": "Managed by host configuration manager."})

def api_node_stop(request):
    port = int(request.GET.get("port", 8080))
    return stop_daemon_ui(request, port)

def api_node_logs(request):
    return JsonResponse({"logs": "Managed by host agent log subsystem."})