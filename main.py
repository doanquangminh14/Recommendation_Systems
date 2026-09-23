"""
Main Entrypoint and CLI Orchestrator for Video Games Recommendation & AI Agent System.

Usage:
    python main.py --mode api       # Start FastAPI REST backend (http://127.0.0.1:8000/docs)
    python main.py --mode ui        # Start Streamlit interactive frontend (http://localhost:8501)
    python main.py --mode test      # Run full automated test suite (Unit, API, UI, E2E)
"""

import os
import sys
import argparse
import subprocess

# Reconfigure stdout for UTF-8 in Windows environments
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BANNER = r"""
========================================================================================
  🎮  AI-POWERED VIDEO GAMES RECOMMENDATION & CONCIERGE SYSTEM
      Hybrid Recommender (SVD + MiniLM-L6 + VADER Sentiment) & AI Agent Platform
========================================================================================
"""


def run_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Start the FastAPI backend server."""
    print(BANNER)
    print(f"🚀 Starting FastAPI REST Endpoints on http://{host}:{port}")
    print(f"📖 OpenAPI Swagger Documentation: http://{host}:{port}/docs")
    print(f"📚 ReDoc Documentation: http://{host}:{port}/redoc\n")
    try:
        import uvicorn
        uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)
    except ImportError:
        print("❌ Error: 'uvicorn' is not installed. Please run: pip install uvicorn")
        sys.exit(1)


def run_ui(port: int = 8501):
    """Launch the interactive Streamlit web application."""
    print(BANNER)
    print(f"🌐 Launching Streamlit Cyber-Gaming UI on http://localhost:{port}...\n")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)


def run_tests():
    """Run all unit and integration test suites."""
    print(BANNER)
    print("🧪 Executing Automated Test Suites (Unit Tests, API Endpoints, UI Components, E2E)...\n")
    
    # 1. Run unittest discover on tests/
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 2. Run API endpoint live tests
    print("\n" + "=" * 60)
    print("🌐 Running Live FastAPI TestClient Tests (11 Endpoints)...")
    print("=" * 60)
    try:
        from tests.test_api import test_api_endpoints
        test_api_endpoints()
    except Exception as e:
        print(f"❌ API test failed: {e}")
        sys.exit(1)

    if result.wasSuccessful():
        print("\n🎉 ALL TEST SUITES PASSED WITH 100% SUCCESS!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)


def run_eval():
    """Run full evaluation metrics and benchmark calculation."""
    print(BANNER)
    print("📈 Running Comprehensive Evaluation & Ranking Benchmarks...\n")
    cmd = [sys.executable, "src/models/evaluation.py"]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="🎮 AI-Powered Video Games Recommendation & Concierge System Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["api", "ui", "test", "eval"],
        default="ui",
        help="Execution mode: 'api' for FastAPI REST server, 'ui' for Streamlit app, 'test' for tests suite, 'eval' for benchmarks report.",
    )
    parser.add_argument("--api", action="store_true", help="Shortcut to start FastAPI server")
    parser.add_argument("--ui", action="store_true", help="Shortcut to start Streamlit UI")
    parser.add_argument("--test", action="store_true", help="Shortcut to run test suites")
    parser.add_argument("--eval", action="store_true", help="Shortcut to run evaluation benchmarks report")
    parser.add_argument("--host", default="127.0.0.1", help="API server host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000 for API, 8501 for UI)")

    args = parser.parse_args()

    if args.eval or args.mode == "eval":
        run_eval()
    elif args.api or args.mode == "api":
        port = args.port or 8000
        run_api(host=args.host, port=port)
    elif args.test or args.mode == "test":
        run_tests()
    else:
        port = args.port or 8501
        run_ui(port=port)


if __name__ == "__main__":
    main()
