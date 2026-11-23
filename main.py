#!/usr/bin/env python3
"""TRACE-X Master Unified Entrypoint.

Usage:
    python main.py                     # Launch Lightweight Web Dashboard (http://localhost:8000)
    python main.py web                 # Launch Lightweight Web Dashboard (http://localhost:8000)
    python main.py api                 # Launch FastAPI Backend Server (http://localhost:8000)
    python main.py live [source]       # Run real-time live video / RTSP / webcam stream
    python main.py analyze <video>     # Run forensic analysis on a CCTV clip
    python main.py reports             # Generate forensic reports for all test clips
    python main.py eval                # Run benchmark evaluations & generate charts
    python main.py test                # Run complete test suite (43 pytest unit tests)
    python main.py gpu                 # Check GPU acceleration & PyTorch CUDA status
    python main.py streamlit           # Launch legacy Streamlit UI (optional)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def cmd_gpu() -> None:
    """Check GPU hardware and PyTorch CUDA acceleration status."""
    import torch
    print("=" * 65)
    print("TRACE-X Hardware & GPU Acceleration Diagnostic")
    print("=" * 65)
    print(f"Python:          {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_avail}")

    if cuda_avail:
        print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
        print(f"Device Count:    {torch.cuda.device_count()}")
        mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"Total VRAM:      {mem_mb:.0f} MB")
        print("\n[Status] Full GPU acceleration is ACTIVE (FP16 Tensor Cores enabled).")
    else:
        print("\n[Status] PyTorch is running in CPU fallback mode.")
        print("Note: To activate your NVIDIA GPU with PyTorch CUDA:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 65)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def cmd_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch TRACE-X Lightweight Web Dashboard (No Streamlit, Instant Launch)."""
    import threading
    import time
    import webbrowser

    active_port = port
    if is_port_in_use(active_port, host):
        print(f"[Notice] Port {active_port} is currently occupied. Searching for next free port...")
        for p in range(active_port + 1, active_port + 20):
            if not is_port_in_use(p, host):
                active_port = p
                break

    print("=" * 65)
    print(f"Launching TRACE-X Web Dashboard on http://localhost:{active_port}")
    print("Zero setup prompts, instant browser launch.")
    print("=" * 65)

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(f"http://localhost:{active_port}")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    subprocess.run([sys.executable, "-m", "uvicorn", "api.app:app", "--host", host, "--port", str(active_port), "--reload"])


def cmd_streamlit(port: int = 8501) -> None:
    """Launch legacy Streamlit UI without email questionnaire."""
    print("=" * 65)
    print(f"Launching legacy Streamlit UI on http://localhost:{port}")
    print("=" * 65)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "ui/app.py",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false"
    ])


def cmd_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch FastAPI Backend Server."""
    print("=" * 65)
    print(f"Launching TRACE-X FastAPI Backend on http://{host}:{port}")
    print("=" * 65)
    subprocess.run([sys.executable, "-m", "uvicorn", "api.app:app", "--host", host, "--port", str(port)])


def cmd_analyze(video_path: str, colour: str | None = None, body: str | None = None) -> None:
    """Analyze a single CCTV video clip."""
    from pipeline.run_pipeline import run_pipeline
    hints = {}
    if colour:
        hints["colour"] = colour
    if body:
        hints["body_type"] = body
    run_pipeline(video_path, hints=hints)


def cmd_live(source: str = "0") -> None:
    """Run real-time live forensic processing on webcam or RTSP stream."""
    from pipeline.live_stream import run_live_stream
    run_live_stream(source=source)


def cmd_reports() -> None:
    """Batch generate forensic reports for all test clips."""
    from scripts.make_all_reports import generate_all_reports
    generate_all_reports()


def cmd_eval() -> None:
    """Run evaluation benchmarks and generate accuracy plots."""
    from eval.run_all import run_all
    run_all()


def cmd_test() -> None:
    """Execute pytest test suite."""
    print("=" * 65)
    print("Running TRACE-X Test Suite...")
    print("=" * 65)
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRACE-X Master Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Web Subcommand (Default)
    web_parser = subparsers.add_parser("web", help="Launch Lightweight Web Dashboard (Default)")
    web_parser.add_argument("--host", type=str, default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000, help="Port to bind dashboard")

    # UI alias to web
    ui_parser = subparsers.add_parser("ui", help="Launch Web Dashboard")
    ui_parser.add_argument("--port", type=int, default=8000, help="Port to bind dashboard")

    # Streamlit Subcommand (Optional legacy)
    st_parser = subparsers.add_parser("streamlit", help="Launch legacy Streamlit UI")
    st_parser.add_argument("--port", type=int, default=8501)

    # API Subcommand
    api_parser = subparsers.add_parser("api", help="Launch FastAPI Backend Server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8000)

    # Analyze Subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a CCTV video clip")
    analyze_parser.add_argument("video", type=str, help="Path to video file")
    analyze_parser.add_argument("--colour", type=str, default=None, help="Colour hint")
    analyze_parser.add_argument("--body", type=str, default=None, help="Body type hint")

    # Live Stream Subcommand
    live_parser = subparsers.add_parser("live", help="Run live video / webcam / RTSP stream")
    live_parser.add_argument("--source", type=str, default="data/clips/jh05ab6799_camA.mp4",
                             help="Camera index ('0'), RTSP URL, or video file path")

    # Reports Subcommand
    subparsers.add_parser("reports", help="Generate forensic reports for all test clips")

    # Eval Subcommand
    subparsers.add_parser("eval", help="Run evaluation suite & produce plots")

    # Test Subcommand
    subparsers.add_parser("test", help="Run pytest unit tests")

    # GPU Subcommand
    subparsers.add_parser("gpu", help="Check GPU status & CUDA diagnostics")

    args = parser.parse_args()

    if args.command in ("web", "ui"):
        cmd_web(port=args.port if hasattr(args, "port") else 8000)
    elif args.command == "streamlit":
        cmd_streamlit(port=args.port)
    elif args.command == "api":
        cmd_api(host=args.host, port=args.port)
    elif args.command == "analyze":
        cmd_analyze(args.video, colour=args.colour, body=args.body)
    elif args.command == "live":
        cmd_live(source=args.source)
    elif args.command == "reports":
        cmd_reports()
    elif args.command == "eval":
        cmd_eval()
    elif args.command == "test":
        cmd_test()
    elif args.command == "gpu":
        cmd_gpu()
    else:
        # Default behavior: if no args, launch the Lightweight Web Dashboard
        print("Launching TRACE-X Web Dashboard on http://localhost:8000...")
        cmd_web()


if __name__ == "__main__":
    main()
