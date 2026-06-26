import os
import socket
import subprocess
import time
import signal
import sys

class LlamaServer:
    def __init__(
        self,
        model_path: str = None,
        mmproj_path: str = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = -1,
        ctx_size: int = 2048,
        binary_path: str = "llama-server",
        thinking: bool = False,
        temperature: float = None,
        top_p: float = None,
        min_p: float = None,
        repeat_penalty: float = None,
        hf_repo: str = None,
        hf_file: str = None,
        hf_mmproj_file: str = None,
        log_dir: str = "/Users/neal/Documents/UI Interface/pipeline/drone_heatmap/logs"
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        
        # --- NEW: Auto-clean UI input traps (strips http:// and trailing slashes) ---
        self.host = host.replace("http://", "").replace("https://", "").split("/")[0] if host else "127.0.0.1"
        self.port = port
        
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.binary_path = binary_path
        self.thinking = thinking
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.repeat_penalty = repeat_penalty
        self.hf_repo = hf_repo
        self.hf_file = hf_file
        self.hf_mmproj_file = hf_mmproj_file
        self.log_dir = log_dir

    def is_port_in_use(self) -> bool:
        """Checks if the target port is already occupied."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((self.host, self.port)) == 0
        except Exception as e:
            print(f"[!] Warning: socket check failed for {self.host}:{self.port} - {e}")
            return False

    def start(self, timeout: int = 45) -> bool:
        """Launches the llama.cpp server inside a visible terminal window."""
        if self.is_port_in_use():
            print(f"[!] Port {self.port} is already in use. Assuming server is alive.")
            return True

        cmd = [
            self.binary_path,
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
        ]

        if self.model_path and os.path.exists(self.model_path):
            cmd.extend(["-m", self.model_path])
        
        if self.mmproj_path and os.path.exists(self.mmproj_path):
            cmd.extend(["--mmproj", self.mmproj_path])

        if self.hf_repo:
            cmd.extend(["--hf-repo", self.hf_repo.strip()])
            if self.hf_file:
                cmd.extend(["--hf-file", self.hf_file.strip()])

        if not (self.model_path and os.path.exists(self.model_path)) and not (self.hf_repo and self.hf_file):
            raise ValueError("Configuration Error: Must provide either a valid model path or Hugging Face repo parameters.")

        if not self.thinking:
            cmd.extend(["--reasoning", "off"])

        if self.temperature is not None: cmd.extend(["--temp", str(self.temperature)])
        if self.top_p is not None: cmd.extend(["--top-p", str(self.top_p)])
        if self.min_p is not None: cmd.extend(["--min-p", str(self.min_p)])
        if self.repeat_penalty is not None: cmd.extend(["--repeat-penalty", str(self.repeat_penalty)])

        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"server_{self.port}.log")

        print(f"[*] Spawning automated llama.cpp repository server on port {self.port}...")

        if sys.platform == "win32":
            log_path = os.path.abspath(log_path)
            cmd_parts = [f'"{arg}"' if " " in arg else arg for arg in cmd]
            raw_cmd = " ".join(cmd_parts)
            command_string = f'cmd /k "{raw_cmd}"'
            subprocess.Popen(command_string, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif sys.platform.startswith("linux"):
            bash_cmd = ["bash", "-c", " ".join(f'"{arg}"' for arg in cmd) + f" 2>&1 | tee '{log_path}'; exec bash"]
            subprocess.Popen(["gnome-terminal", "--"] + bash_cmd)
        else: 
            cmd_str = " ".join(f"'{arg}'" for arg in cmd) + f" 2>&1 | tee '{log_path}'"
            escaped_cmd_str = cmd_str.replace('"', '\\"')
            applescript_cmd = f'tell application "Terminal" to do script "{escaped_cmd_str}"'
            subprocess.Popen(["osascript", "-e", applescript_cmd])

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_port_in_use():
                print(f"[+] Server successfully online at http://{self.host}:{self.port}")
                return True
            time.sleep(0.5)

        print("[-] Server startup timed out.")
        return False

    def stop(self):
        """Surgically terminates the server binary process and closes its macOS Terminal window container."""
        print(f"[*] Shutting down llama.cpp server on port {self.port}...")
        
        if sys.platform == "win32":
            try:
                netstat_cmd = f"netstat -ano | findstr :{self.port}"
                output = subprocess.check_output(netstat_cmd, shell=True, text=True)
                pids = set()
                for line in output.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5 and "LISTENING" in line:
                        pids.add(parts[-1])
                for pid in pids:
                    subprocess.run(["taskkill", "/F", "/PID", pid], check=True)
                print("[+] Server process killed cleanly.")
            except subprocess.CalledProcessError:
                print(f"[!] No active server found on port {self.port}.")
                
        elif sys.platform.startswith("linux"):
            try:
                pid_cmd = f"fuser {self.port}/tcp"
                pid_output = subprocess.check_output(pid_cmd, shell=True, text=True).strip()
                if pid_output:
                    pids = pid_output.split()
                    for pid in pids:
                        os.kill(int(pid), signal.SIGTERM)
                    print("[+] Server process killed cleanly.")
            except subprocess.CalledProcessError:
                print(f"[!] No active server found on port {self.port}.")
                
        else:
            try:
                pid_cmd = ["lsof", "-t", f"-i:{self.port}"]
                pid_output = subprocess.check_output(pid_cmd, text=True).strip()
                if pid_output:
                    pids = pid_output.split("\n")
                    for pid in pids:
                        if pid.strip():
                            os.kill(int(pid.strip()), signal.SIGKILL)
                    print("[+] Server process killed cleanly.")
                
                applescript_close = f'''
                tell application "Terminal"
                    repeat with w in windows
                        try
                            set tabList to tabs of w
                            repeat with t in tabList
                                if history of t contains "--port {self.port}" then
                                    close w
                                    exit repeat
                                end if
                            end repeat
                        catch
                        end try
                    </repeat>
                end tell
                '''
                subprocess.Popen(["osascript", "-e", applescript_close])
                print("[+] Associated macOS Terminal window closed cleanly.")
            except subprocess.CalledProcessError:
                print(f"[!] No active server found on port {self.port}.")
            except Exception as e:
                print(f"[-] Error stopping server: {e}")