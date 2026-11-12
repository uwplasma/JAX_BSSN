"""Exercise CPU collectives across two independent JAX runtime processes."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest


class TestFMRProcesses(unittest.TestCase):
    def test_two_process_interpolation_injection_and_synchronized_step(self):
        root = Path(__file__).resolve().parents[1]
        worker = root / "tests" / "support" / "fmr_process_worker.py"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            address = "127.0.0.1:" + str(listener.getsockname()[1])

        env = os.environ.copy()
        env.update(
            JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="1", JAX_ENABLE_X64="true",
            JAX_CPU_COLLECTIVES_IMPLEMENTATION="gloo", GLOO_SOCKET_IFNAME="lo",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            PYTHONUNBUFFERED="1", PYTHONPATH=str(root),
        )
        # Only these loopback workers need proxy-free networking. Never alter
        # the parent shell or the user's environment for production jobs.
        for name in list(env):
            if name.lower().endswith("_proxy"):
                env.pop(name)
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []

        with tempfile.TemporaryDirectory(prefix="bssn-fmr-process-test-") as temporary:
            processes, streams, paths = [], [], []
            try:
                for rank in range(2):
                    path = Path(temporary) / f"worker-{rank}.log"
                    stream = path.open("w")
                    rank_env = env.copy()
                    cpus = available[rank::2][:4]
                    if cpus:
                        rank_env["BSSN_TEST_CPU_AFFINITY"] = ",".join(map(str, cpus))
                    streams.append(stream)
                    paths.append(path)
                    processes.append(subprocess.Popen(
                        [sys.executable, str(worker), address, str(rank)],
                        cwd=root, env=rank_env, stdout=stream, stderr=subprocess.STDOUT,
                    ))

                deadline = time.monotonic() + 600
                while any(process.poll() is None for process in processes):
                    if any(process.returncode not in (None, 0) for process in processes):
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.1)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                for stream in streams:
                    stream.close()

            logs = [path.read_text() for path in paths]
            report = "\n".join(f"worker {rank}:\n{log}" for rank, log in enumerate(logs))
            self.assertEqual([process.returncode for process in processes], [0, 0], report)
            results = []
            for log in logs:
                records = [line.removeprefix("BSSN_FMR_DISTRIBUTED_RESULT ")
                           for line in log.splitlines()
                           if line.startswith("BSSN_FMR_DISTRIBUTED_RESULT ")]
                self.assertEqual(len(records), 1, report)
                results.append(json.loads(records[0]))
            self.assertEqual({result["process_id"] for result in results}, {0, 1})
            for result in results:
                self.assertEqual(result["process_count"], 2)
                self.assertEqual(result["local_device_count"], 1)
                self.assertLess(result["halo_error"], 1.0e-11)
                self.assertLess(result["injection_error"], 1.0e-12)
                self.assertLess(result["evolution_error"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
