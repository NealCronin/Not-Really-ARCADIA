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
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = -1,
        ctx_size: int = 2048,
        binary_path: str = "llama-server",
        cpu_moe: int = 0,
        flash_attn: bool = False,
        cache_type_k: str = "",
        cache_type_v: str = "",
        spec_type: str = "",
        spec_draft_n_max: int = 0,
        spec_draft_p_min: float = 0.0,
        spec_type_secondary: str = "",
        spec_ngram_mod_n_match: int = 0,
        jinja: bool = False,
        chat_template_kwargs: str = "",
        reasoning_budget: int = 0,
        reasoning_budget_message: str = "",
        temperature: float = None,
        top_p: float = None,
        min_p: float = None,
        top_k: int = None,
        presence_penalty: float = None,
        repeat_penalty: float = None,
        hf_repo: str = None,
        hf_file: str = None,
        hf_mmproj: str = None,
        log_dir: str = None
    ):
        self.model_path = model_path
        self.host = host.replace("http://", "").replace("https://", "").split("/")[0] if host else "127.0.0.1"
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.binary_path = binary_path
        
        # Advanced Overrides
        self.cpu_moe = cpu_moe
        self.flash_attn = flash_attn
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.spec_type = spec_type
        self.spec_draft_n_max = spec_draft_n_max
        self.spec_draft_p_min = spec_draft_p_min
        self.spec_type_secondary = spec_type_secondary
        self.spec_ngram_mod_n_match = spec_ngram_mod_n_match
        self.jinja = jinja
        self.chat_template_kwargs = chat_template_kwargs
        self.reasoning_budget = reasoning_budget
        self.reasoning_budget_message = reasoning_budget_message

        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repeat_penalty = repeat_penalty
        
        self.hf_repo = hf_repo
        self.hf_file = hf_file
        self.hf_mmproj = hf_mmproj
        self.log_dir = log_dir

    def is_port_in_use(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                return s.connect_ex((self.host, self.port)) == 0
        except socket.error:
            return False
        except Exception:
            return False

    def start(self, timeout: int = 45) -> bool:
        if self.is_port_in_use(): return True

        cmd = [self.binary_path, "--host", self.host, "--port", str(self.port), "-ngl", str(self.n_gpu_layers), "-c", str(self.ctx_size)]

        if self.model_path and os.path.exists(self.model_path): cmd.extend(["-m", self.model_path])
        
        # Hugging Face integration
        if self.hf_repo:
            cmd.extend(["--hf-repo", self.hf_repo.strip()])
            if self.hf_file: cmd.extend(["--hf-file", self.hf_file.strip()])
            if self.hf_mmproj: cmd.extend(["--hf-mmproj-file", self.hf_mmproj.strip()])

        if not (self.model_path and os.path.exists(self.model_path)) and not (self.hf_repo and self.hf_file):
            raise ValueError("Configuration Error: Must provide either a valid model path or Hugging Face repo.")

        # Advanced Architecture Flags
        if self.cpu_moe > 0: cmd.extend(["--n-cpu-moe", str(self.cpu_moe)])
        if self.flash_attn: cmd.append("--flash-attn")
        if self.cache_type_k: cmd.extend(["--cache-type-k", self.cache_type_k.strip()])
        if self.cache_type_v: cmd.extend(["--cache-type-v", self.cache_type_v.strip()])
        
        # Speculative Decoding
        if self.spec_type: cmd.extend(["--spec-type", self.spec_type.strip()])
        if self.spec_draft_n_max > 0: cmd.extend(["--spec-draft-n-max", str(self.spec_draft_n_max)])
        if self.spec_draft_p_min > 0: cmd.extend(["--spec-draft-p-min", str(self.spec_draft_p_min)])
        if self.spec_type_secondary: cmd.extend(["--spec-type-secondary", self.spec_type_secondary.strip()])
        if self.spec_ngram_mod_n_match > 0: cmd.extend(["--spec-ngram-mod-n-match", str(self.spec_ngram_mod_n_match)])
        
        # Reasoning & Templates
        if self.jinja: cmd.append("--jinja")
        if self.chat_template_kwargs: cmd.extend(["--chat-template-kwargs", self.chat_template_kwargs.strip()])
        if self.reasoning_budget > 0: cmd.extend(["--reasoning-budget", str(self.reasoning_budget)])
        if self.reasoning_budget_message: cmd.extend(["--reasoning-budget-message", self.reasoning_budget_message.strip()])

        # Sampling
        if self.temperature is not None: cmd.extend(["--temp", str(self.temperature)])
        if self.top_p is not None: cmd.extend(["--top-p", str(self.top_p)])
        if self.min_p is not None: cmd.extend(["--min-p", str(self.min_p)])
        if self.top_k is not None: cmd.extend(["--top-k", str(self.top_k)])
        if self.presence_penalty is not None: cmd.extend(["--presence-penalty", str(self.presence_penalty)])
        if self.repeat_penalty is not None: cmd.extend(["--repeat-penalty", str(self.repeat_penalty)])

        os.makedirs(self.log_dir, exist_ok=True) if self.log_dir else None
        log_path = os.path.join(self.log_dir, f"server_{self.port}.log") if self.log_dir else None

        if sys.platform == "win32":
            if log_path:
                log_path = os.path.abspath(log_path)
                cmd_parts = [f'"{arg}"' if " " in arg else arg for arg in cmd]
                raw_cmd = " ".join(cmd_parts)
                command_string = f'cmd /k "{raw_cmd}"'
                subprocess.Popen(command_string, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif sys.platform.startswith("linux"):
            if log_path:
                bash_cmd = ["bash", "-c", " ".join(f'"{arg}"' for arg in cmd) + f" 2>&1 | tee '{log_path}'; exec bash"]
                subprocess.Popen(["gnome-terminal", "--"] + bash_cmd)
            else:
                subprocess.Popen(["gnome-terminal", "--"] + cmd)
        else: 
            if log_path:
                cmd_str = " ".join(f"'{arg}'" for arg in cmd) + f" 2>&1 | tee '{log_path}'"
                escaped_cmd_str = cmd_str.replace('"', '\\"')
                applescript_cmd = f'tell application "Terminal" to do script "{escaped_cmd_str}"'
                subprocess.Popen(["osascript", "-e", applescript_cmd])
            else:
                subprocess.Popen(["open", "-a", "Terminal.app"] + cmd)

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_port_in_use(): return True
            time.sleep(0.5)

        return False

    def stop(self):
        if sys.platform == "win32":
            try:
                output = subprocess.check_output(f"netstat -ano | findstr :{self.port}", shell=True, text=True)
                for line in output.strip().split("\n"):
                    if "LISTENING" in line: subprocess.run(["taskkill", "/F", "/PID", line.split()[-1]], check=True)
            except: pass
        elif sys.platform.startswith("linux"):
            try:
                output = subprocess.check_output(f"fuser {self.port}/tcp", shell=True, text=True).strip()
                if output:
                    for pid in output.split(): os.kill(int(pid), signal.SIGTERM)
            except: pass
        else:
            try:
                output = subprocess.check_output(["lsof", "-t", f"-i:{self.port}"], text=True).strip()
                if output:
                    for pid in output.split("\n"):
                        if pid.strip(): os.kill(int(pid.strip()), signal.SIGKILL)
                applescript_close = f'tell application "Terminal"\nrepeat with w in windows\ntry\nrepeat with t in tabs of w\nif history of t contains "--port {self.port}" then\nclose w\nexit repeat\nend if\nend repeat\ncatch\nend try\nend repeat\nend tell'
                subprocess.Popen(["osascript", "-e", applescript_close])
            except: pass