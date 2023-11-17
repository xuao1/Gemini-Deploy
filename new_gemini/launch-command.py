#!/usr/bin/env python3
import os
import subprocess as sp
import argparse
import signal
import time
import warnings
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, help='seconds to run')
    parser.add_argument('command', nargs='+')
    args = parser.parse_args()

    os.setpgrp()

    client_env = os.environ.copy()
    client_env['LD_PRELOAD'] = "{}/Gemini-Deploy/new_gemini/lib/libgemhook.so.1".format(Path.home())

    proc = sp.Popen(
        args.command, env=client_env, start_new_session=True, universal_newlines=True, bufsize=1, shell=True, stderr=sp.STDOUT
    )

    print("[launcher] run: {}".format(args.command))

    try:
        if args.timeout:
            time.sleep(args.timeout)
            proc.terminate()
            proc.wait()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError as e:
                warnings.warn(e)
        else:
            proc.wait()
    except KeyboardInterrupt:
        print("\n[launcher] kill everything")
        os.killpg(proc.pid, signal.SIGTERM)
        os.killpg(0, signal.SIGTERM)


if __name__ == '__main__':
    main()
