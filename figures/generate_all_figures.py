import subprocess
import sys
import os

figures_dir = os.path.dirname(os.path.abspath(__file__))
figure_scripts = [
    "generate_fig1.py",
    "generate_fig2.py",
    "generate_fig3.py",
    "generate_fig4.py",
    "generate_fig5.py",
    "generate_fig6.py"
]

print("Starting full figure generation pipeline...")
for script in figure_scripts:
    script_path = os.path.join(figures_dir, script)
    if not os.path.exists(script_path):
        print(f"Error: {script} not found at {script_path}. Skipping.")
        continue
    
    print(f"\n---> Running {script}...")
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Finished {script} successfully.")
        if res.stdout:
            print(res.stdout.strip())
    else:
        print(f"Error: {script} failed with exit code {res.returncode}.")
        if res.stderr:
            print(res.stderr.strip())

print("\nAll figures completed.")
