import subprocess
print "`python main.py` will be deprecated. Please use `./run.sh` instead."
cmd = "./run.sh"
subprocess.call(cmd, shell=True)
