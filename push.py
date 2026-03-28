import subprocess
import time

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode

run("git init")
run("git add .")
run('git commit -m "model refined and switched UI to html css javascript"')
run("git branch -M main")
run("git remote remove origin")
run("git remote add origin https://github.com/Vo1denz/insurance-fraud-detection-using-ML.git")
run("git push -u origin main")
