"""NTRO Social Media Analytics — Master Demo Launcher.

Launches both FastAPI backend server and React Vite dashboard concurrently.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


def main():
    parser = argparse.ArgumentParser(
        description="Launch NTRO Social Media Analytics FastAPI backend and React frontend."
    )
    parser.add_argument("--backend-port", type=int, default=8000, help="FastAPI backend port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, default=5173, help="Vite frontend port (default: 5173)")
    parser.add_argument("--backend-only", action="store_true", help="Launch only FastAPI backend")
    parser.add_argument("--frontend-only", action="store_true", help="Launch only React frontend")

    args = parser.parse_args()

    processes = []

    def shutdown(sig=None, frame=None):
        print("\n\033[1;33m[!] Stopping NTRO Social Media Analytics services...\033[0m")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                p.kill()
        print("\033[1;32m[OK] All services stopped cleanly.\033[0m")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\033[1;36m" + "=" * 65)
    print("      NTRO SOCIAL MEDIA ANALYTICS — INTELLIGENCE DASHBOARD")
    print("=" * 65 + "\033[0m")

    # 1. Start FastAPI Backend
    if not args.frontend_only:
        print(f"\033[1;34m[*] Starting FastAPI backend on http://localhost:{args.backend_port}...\033[0m")
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(args.backend_port),
            "--reload",
        ]
        p_backend = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT)
        processes.append(p_backend)

    # 2. Start Vite Frontend
    if not args.backend_only:
        print(f"\033[1;34m[*] Starting React Vite frontend on http://localhost:{args.frontend_port}...\033[0m")
        # On Windows, npm is a cmd script
        npm_exec = "npm.cmd" if sys.platform == "win32" else "npm"
        frontend_cmd = [npm_exec, "run", "dev", "--", "--port", str(args.frontend_port)]
        p_frontend = subprocess.Popen(frontend_cmd, cwd=FRONTEND_DIR)
        processes.append(p_frontend)

    time.sleep(1)

    print("\033[1;32m" + "-" * 65)
    print("  [OK] Services Active:")
    if not args.backend_only:
        print(f"      - Intelligence Dashboard UI:  \033[1;37mhttp://localhost:{args.frontend_port}\033[0m")
    if not args.frontend_only:
        print(f"      - FastAPI Backend API:        \033[1;37mhttp://localhost:{args.backend_port}\033[0m")
        print(f"      - Interactive Swagger Docs:   \033[1;37mhttp://localhost:{args.backend_port}/docs\033[0m")
    print("-" * 65 + "\033[0m")
    print("\033[0;33mPress Ctrl+C to terminate all services.\033[0m\n")

    try:
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"\033[1;31m[!] Process {p.pid} exited with code {p.returncode}.\033[0m")
                    shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
