"""Serialize vLLM startup/warmup with other models on the shared-GPU profile.

Inference is serialized by the generation API. This wrapper additionally covers
model-server restart, when there is no API request around vLLM's warmup work.
"""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    child, stopped = None, False

    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    handle = None
    try:
        if os.environ.get('GPU_SERIALIZE', '1') != '0':
            path = Path(os.environ.get('GPU_LOCK_PATH', '/gpu-lock/inference.lock'))
            handle = path.open('a+')
            deadline = time.monotonic() + 3600
            while True:
                if stopped:
                    return 0
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('GPU startup lock wait exceeded 1 hour')
                    time.sleep(0.1)
        child = subprocess.Popen(['vllm', 'serve', *sys.argv[1:]], start_new_session=True)
        deadline = time.monotonic() + 3600
        ready = False
        while child.poll() is None and not stopped:
            try:
                req = urllib.request.Request('http://127.0.0.1:8000/health',
                    headers={'Authorization': 'Bearer ' + os.environ.get('API_KEY', '')})
                with urllib.request.urlopen(req, timeout=3) as response:
                    ready = response.status == 200
                if ready:
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                raise TimeoutError('vLLM did not become ready within 1 hour')
            time.sleep(1)
        if child.poll() is not None:
            return child.returncode
    finally:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
    return child.wait()


if __name__ == '__main__':
    raise SystemExit(main())
