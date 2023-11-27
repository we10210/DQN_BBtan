import subprocess

# Run the other script
for run in range(5):
    print(f"第{run + 1}個循環")
    subprocess.run(["python", "dqn.py"])