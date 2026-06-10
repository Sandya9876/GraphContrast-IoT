import subprocess, sys
subprocess.run([sys.executable, 'main.py', '--config', 'configs/wadi.yaml'], check=True)
