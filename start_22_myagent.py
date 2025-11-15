#!/usr/bin/env python3
"""
22_MyAgent System Launcher
Starts all components of the Continuous AI App Builder
"""

import os
import subprocess
import signal
import time
import sys
from pathlib import Path
import psutil
from loguru import logger

class MyAgentLauncher:
    def __init__(self):
        self.base_dir = Path('/home/aadel/projects/22_MyAgent')
        self.processes = {}
        self.running = True
        
    def log_info(self, message):
        print(f"🤖 {message}")
        
    def log_success(self, message):
        print(f"✅ {message}")
        
    def log_error(self, message):
        print(f"❌ {message}")
        
    def check_port(self, port):
        """Check if port is available"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return False
        return True
        
    def kill_port(self, port):
        """Kill process using port"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        proc.terminate()
                        self.log_info(f"Killed process using port {port}")
                        time.sleep(1)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
        
    def setup_environment(self):
        """Setup environment and check prerequisites"""
        self.log_info("Setting up environment...")
        
        # Change to project directory
        os.chdir(self.base_dir)
        
        # Check if virtual environment exists
        venv_path = self.base_dir / 'venv' / 'bin' / 'activate'
        if not venv_path.exists():
            self.log_error("Virtual environment not found")
            return False
            
        # Set environment variables
        os.environ['PYTHONPATH'] = str(self.base_dir)
        
        self.log_success("Environment setup complete")
        return True
        
    def start_api_server(self):
        """Start FastAPI server"""
        self.log_info("Starting API server...")
        
        # Kill existing process on port 8000
        if not self.check_port(8000):
            self.kill_port(8000)
            
        cmd = [
            'bash', '-c',
            f'cd {self.base_dir} && source venv/bin/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000'
        ]
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            self.processes['api'] = process
            
            # Wait a moment and check if it started
            time.sleep(3)
            if process.poll() is None:
                self.log_success("API server started on http://136.112.55.140:8000")
                return True
            else:
                self.log_error("API server failed to start")
                return False
        except Exception as e:
            self.log_error(f"Failed to start API server: {e}")
            return False
            
    def start_frontend(self):
        """Start React frontend"""
        self.log_info("Starting frontend dashboard...")
        
        # Kill existing process on port 5173
        if not self.check_port(5173):
            self.kill_port(5173)
            
        cmd = [
            'bash', '-c',
            f'cd {self.base_dir}/frontend && npm run dev -- --host 0.0.0.0 --port 5173'
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            self.processes['frontend'] = process
            
            # Wait a moment and check if it started
            time.sleep(5)
            if process.poll() is None:
                self.log_success("Frontend started on http://136.112.55.140:5173")
                return True
            else:
                self.log_error("Frontend failed to start")
                return False
        except Exception as e:
            self.log_error(f"Failed to start frontend: {e}")
            return False
            
    def start_orchestrator(self):
        """Start continuous orchestrator"""
        self.log_info("Starting continuous orchestrator...")
        
        cmd = [
            'bash', '-c', 
            f'cd {self.base_dir} && source venv/bin/activate && python -m core.orchestrator.continuous_director'
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            self.processes['orchestrator'] = process
            
            # Wait a moment and check if it started
            time.sleep(2)
            if process.poll() is None:
                self.log_success("Continuous orchestrator started")
                return True
            else:
                self.log_error("Orchestrator failed to start")
                return False
        except Exception as e:
            self.log_error(f"Failed to start orchestrator: {e}")
            return False
            
    def monitor_processes(self):
        """Monitor all processes"""
        while self.running:
            try:
                time.sleep(5)
                
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        self.log_error(f"{name} process died, exit code: {process.returncode}")
                        # Optionally restart the process
                        # self.restart_process(name)
                        
            except KeyboardInterrupt:
                self.log_info("Received shutdown signal...")
                self.shutdown()
                break
                
    def shutdown(self):
        """Gracefully shutdown all processes"""
        self.log_info("Shutting down 22_MyAgent system...")
        self.running = False
        
        for name, process in self.processes.items():
            self.log_info(f"Stopping {name}...")
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                self.log_success(f"{name} stopped")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop gracefully
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                self.log_info(f"Force killed {name}")
            except Exception as e:
                self.log_error(f"Error stopping {name}: {e}")
                
        self.log_success("22_MyAgent system shutdown complete")
        
    def start_all(self):
        """Start all components"""
        self.log_info("🚀 Starting 22_MyAgent Continuous AI App Builder...")
        
        if not self.setup_environment():
            return False
            
        # Start components in order
        components = [
            ("API Server", self.start_api_server),
            ("Frontend Dashboard", self.start_frontend), 
            ("Continuous Orchestrator", self.start_orchestrator)
        ]
        
        for name, start_func in components:
            if not start_func():
                self.log_error(f"Failed to start {name}, aborting...")
                self.shutdown()
                return False
                
        # Print access information
        print("\n" + "="*60)
        print("🎉 22_MYAGENT SYSTEM FULLY OPERATIONAL")
        print("="*60)
        print("📍 Access URLs:")
        print("   🌐 Frontend Dashboard: http://136.112.55.140:5173")
        print("   🔧 API Server: http://136.112.55.140:8000")
        print("   📚 API Documentation: http://136.112.55.140:8000/docs")
        print("   📊 Real-time WebSocket: ws://136.112.55.140:8000/ws/PROJECT_ID")
        print()
        print("🤖 System Components Running:")
        print("   ✅ FastAPI Backend (Port 8000)")
        print("   ✅ React Frontend (Port 5173)") 
        print("   ✅ Continuous AI Orchestrator")
        print()
        print("🎯 Ready for continuous AI development!")
        print("   Press Ctrl+C to shutdown gracefully")
        print("="*60)
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            pass
        finally:
            if self.running:
                self.shutdown()
                
        return True

def main():
    """Main launcher function"""
    launcher = MyAgentLauncher()
    
    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        launcher.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the system
    success = launcher.start_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
