import subprocess
import sys
import time

PIPELINE_STEPS = [
    ("Preprocessing", "cancerpreprocess.py"),
    ("Spectral Analysis", "SpectralAnalysis.py"),
    ("TDA", "TDACancer.py")
]

print("=" * 60, flush=True)
print("STARTING PIPELINE", flush=True)
print("=" * 60, flush=True)

start = time.time()

for name, script in PIPELINE_STEPS:

    print(f"\nRUNNING: {name}", flush=True)

    process = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # STREAM CHILD OUTPUT
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip(), flush=True)

    process.stdout.close()

    process.wait()

    if process.returncode != 0:

        print(f"FAILED: {script}", flush=True)
        exit(1)

    print(f"COMPLETED: {script}", flush=True)

elapsed = time.time() - start

print("\nPIPELINE FINISHED", flush=True)
print(f"Total time: {elapsed:.2f}s", flush=True)