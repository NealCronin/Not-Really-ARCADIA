import socket
import threading

class VPNTunnelBridge:
    """A generalized background network socket proxy that pipes local port calls through a VPN tunnel."""
    def __init__(self, local_port: int, remote_host: str, remote_port: int):
        self.local_host = "127.0.0.1"
        self.local_port = local_port
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.running = False
        self.server_socket = None
        self.traffic_log = []

    def log(self, message: str):
        self.traffic_log.append(f"[{threading.current_thread().name}] {message}")
        if len(self.traffic_log) > 40:
            self.traffic_log.pop(0)

    def _pipe(self, source, destination):
        try:
            while self.running:
                data = source.recv(262144)
                if not data: 
                    break
                destination.sendall(data)
        except: 
            pass
        finally:
            # Signal to the destination that we are done writing data,
            # but keep it open so the other thread can still read from it!
            try: 
                destination.shutdown(socket.SHUT_WR)
            except: 
                pass
            try: 
                source.shutdown(socket.SHUT_RD)
            except: 
                pass

    def _handle_client(self, client_socket, addr):
        self.log(f"Connection from local client {addr[0]}:{addr[1]}")
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            remote_socket.connect((self.remote_host, self.remote_port))
            self.log(f"Tunnel routing secured ➡️ [{self.remote_host}]:{self.remote_port}")
        except Exception as e:
            self.log(f"Tunnel routing link failed: {e}")
            client_socket.close()
            return

        # Helper function to close everything only when BOTH threads are done
        def clean_close():
            try: client_socket.close()
            except: pass
            try: remote_socket.close()
            except: pass

        t1 = threading.Thread(target=self._pipe, args=(client_socket, remote_socket), daemon=True)
        t2 = threading.Thread(target=self._pipe, args=(remote_socket, client_socket), daemon=True)
        
        t1.start()
        t2.start()

    def start(self) -> bool:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.local_host, self.local_port))
            self.server_socket.listen(100)
            self.running = True
            self.log(f"Proxy bridge bound on local port {self.local_port}")
        except Exception as e:
            return False
        def listen_loop():
            while self.running:
                try:
                    client_conn, addr = self.server_socket.accept()
                    self._handle_client(client_conn, addr)
                except: break
        threading.Thread(target=listen_loop, name=f"BridgeProxy-{self.local_port}", daemon=True).start()
        return True

    def stop(self):
        self.running = False
        if self.server_socket:
            try: self.server_socket.close()
            except: pass