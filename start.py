import sys
import subprocess
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).parent
VENV_DIR = PROJECT_DIR / "venv"

def run(cmd):
    subprocess.check_call(cmd)

def venv_python():
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def ensure_venv():
    if not VENV_DIR.exists():
        print("[+] Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])

def ensure_requirements():
    python = venv_python()
    print("[+] Installing requirements...")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-r", "requirements.txt"])

def run_streamlit():
    python = venv_python()
    print("[+] Starting Streamlit...")
    run([str(python), "-m", "streamlit", "run", "app/ui.py"])

if __name__ == "__main__":
    ensure_venv()
    ensure_requirements()
    run_streamlit()
