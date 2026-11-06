"""Run the authorized RQ4 repair and its artifact audit to completion."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rq4_release_repair_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (OUT/'supervisor.pid').open('w') as f:
        f.write(str(os.getpid()))
    # Prevent idle system sleep while this process runs (does not prevent lid sleep).
    awake = subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        for command in [
            [sys.executable,'-u','experiments/rq4_release_repair.py','--scope','all'],
            [sys.executable,'-u','analysis/audit_rq4_repair.py']]:
            subprocess.run(command, cwd=ROOT, check=True)
    finally:
        awake.terminate()
        print(f'Elapsed supervisor wall seconds: {time.time()-started:.1f}', flush=True)


if __name__ == '__main__':
    main()
