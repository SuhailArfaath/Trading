"""Kill any existing server on port 8500, then start the FastAPI server."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Find and kill process listening on port 8500
r = subprocess.run(
    'netstat -ano | findstr ":8500 " | findstr LISTENING',
    shell=True, capture_output=True, text=True,
)
if r.stdout.strip():
    pid = r.stdout.strip().split()[-1]
    subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
    print(f"Killed PID {pid}")

# Start server (absolute path, so cwd does not matter)
sys.exit(subprocess.run([sys.executable, str(HERE / 'api.py')]).returncode)