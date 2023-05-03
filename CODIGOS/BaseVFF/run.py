import os
import glob

base_path = os.getcwd()

files = glob.glob("*.py")

for f in files:
    if not f == "run.py":
        print(f"Running file: {f}")
        os.system("python %s"%(f))
