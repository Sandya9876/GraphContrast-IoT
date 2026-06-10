import subprocess, sys
subprocess.run([sys.executable, 'main.py', '--config', 'configs/swat.yaml'], check=True)
