import subprocess, sys
subprocess.run([sys.executable, 'main.py', '--config', 'configs/skab.yaml'], check=True)
