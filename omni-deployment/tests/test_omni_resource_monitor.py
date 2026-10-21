"""CPU-only parser, process-scope and termination evidence checks; no GPU needed."""
import argparse
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import monitor_omni_resources as monitor


def proc_stat(ppid=1, start=100):
    fields = ["S", str(ppid)] + ["0"] * 17 + [str(start)]
    return "10 (worker name (nested)) " + " ".join(fields)


class MonitorTests(unittest.TestCase):
    def test_reject_nonfinite_or_nonpositive_options(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                monitor.positive_number(value)
        self.assertEqual(monitor.positive_number("0.5"), 0.5)
        with self.assertRaises(argparse.ArgumentTypeError):
            monitor.positive_pid("0")

    def test_gpu_csv_units_missing_values_and_invalid_rows(self):
        rows = monitor.parse_gpu_csv('0, GPU-a, "RTX, 5090", 23, 10, 2048, 32768\n1, GPU-b, RTX 5090, [N/A], 0, 4096, 32768\n')
        self.assertEqual(rows[0]["name"], "RTX, 5090")
        self.assertEqual(rows[1]["memory_used_mib"], 4096)
        self.assertIsNone(rows[1]["utilization_gpu_percent"])
        for text in ("", "0,bad", "0,GPU-a,5090,101,0,2,3", "0,GPU-a,5090,0,0,nan,3"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                monitor.parse_gpu_csv(text)

    def test_meminfo_and_stat_parsers(self):
        result = monitor.parse_meminfo("MemTotal: 1000 kB\nMemAvailable: 600 kB\nSwapTotal: 300 kB\nSwapFree: 200 kB\nIgnored: 99 kB\n")
        self.assertEqual(result["mem_available_bytes"], 600 * 1024)
        self.assertEqual(monitor.parse_proc_stat(proc_stat(99, 123))["start_ticks"], 123)
        with self.assertRaises(ValueError):
            monitor.parse_meminfo("MemTotal: 1000 kB\n")

    def test_scope_and_pid_reuse_do_not_include_unrelated_ray(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = Path("/home/isaac/workdir/h3")
            def add(pid, argv, ppid=1, start=100):
                proc = root / str(pid)
                proc.mkdir()
                (proc / "stat").write_text(proc_stat(ppid, start))
                (proc / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
                (proc / "status").write_text("VmRSS:\t100 kB\n")
                (proc / "environ").write_bytes(b"MUST_NOT_READ=secret")
                return proc
            add(10, [str(project / "envs/omni/bin/python"), "ignored-secret-argument"])
            add(11, ["ray::RayWorkerWrapper.execute_method"], ppid=10)
            add(12, ["ray::RayWorkerWrapper.execute_method"])
            add(13, [str(project / "envs/omni-other/bin/python"), "/unrelated.py"])
            add(14, ["/usr/bin/python3", "-u", str(project / "omni-deployment/scripts/serve.py"), "ignored-secret-argument"])
            add(15, ["ray::RayWorkerWrapper.execute_method"], start=500)
            rows, diagnostics = monitor.process_snapshot(root, project, {15: 500})
            self.assertEqual({row["pid"] for row in rows}, {10, 11, 14, 15})
            self.assertEqual(diagnostics["selected_processes_without_rss"], [])
            self.assertNotIn("secret", json.dumps(rows))
            # An explicit PID is a different process after its start time changes.
            (root / "15/stat").write_text(proc_stat(start=501))
            rows, _ = monitor.process_snapshot(root, project, {15: 500})
            self.assertNotIn(15, {row["pid"] for row in rows})
            # RSS failure is unavailable evidence, not a zero-RSS process.
            (root / "10/status").write_text("VmRSS: N/A\n")
            rows, diagnostics = monitor.process_snapshot(root, project, {})
            self.assertNotIn(10, {row["pid"] for row in rows})
            self.assertEqual(diagnostics["selected_processes_without_rss"][0]["pid"], 10)

    def test_prefix_reader_stops_before_application_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            stream = io.BytesIO(b"/usr/bin/python3\0/home/h3/omni-deployment/scripts/main.py\0credential=do-not-read\0")
            from unittest.mock import patch
            with patch.object(Path, "open", return_value=stream):
                # BytesIO closes on exit; inspect its position at close instead.
                position = []
                original_close = stream.close
                stream.close = lambda: position.append(stream.tell())
                self.assertEqual(monitor.process_scope(proc, Path("/home/h3")), "project_script")
                self.assertEqual(position, [len(b"/usr/bin/python3\0/home/h3/omni-deployment/scripts/main.py\0")])
                original_close()

    def test_signal_stop_writes_jsonl_and_companion_peaks(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary) / "samples.jsonl", project_root=Path("/home/h3"),
                                   interval=0.001, duration=None)
            stop = monitor.Stop()
            calls = []
            def sampler(_args, _actors):
                calls.append(1)
                second = len(calls) == 2
                if second:
                    stop(signal.SIGTERM)
                return {"kind": "sample", "gpus": monitor.parse_gpu_csv("0,GPU-a,5090,45,10,1100,32768" if second else "0,GPU-a,5090,45,10,1200,32768"),
                        "host": dict(mem_total_bytes=1000, mem_available_bytes=300 if second else 400, swap_total_bytes=100, swap_free_bytes=70),
                        "processes": [dict(pid=50, start_ticks=7, rss_bytes=100 if second else 120, scope="project_venv", project_ancestor_pid=None)],
                        "errors": [], "selected_process_rss_sum_bytes": 100 if second else 120, "sampling_duration_s": 0.01}
            result = monitor.monitor(args, {}, stop, sampler=sampler)
            self.assertEqual(result["stop_reason"], "signal")
            self.assertEqual(result["stop_signal"], signal.SIGTERM)
            self.assertEqual(result["gpus"]["GPU-a"]["peak_memory_used_mib"], 1200)
            self.assertEqual(result["host"]["min_mem_available_bytes"], 300)
            self.assertEqual(result["host"]["peak_swap_used_bytes"], 30)
            self.assertEqual(result["processes"]["50:7"]["peak_rss_bytes"], 120)
            self.assertEqual(json.loads(args.output.with_suffix(".summary.json").read_text()), result)
            lines = [json.loads(line) for line in args.output.read_text().splitlines()]
            self.assertEqual([line["kind"] for line in lines], ["metadata", "sample", "sample", "summary"])
            with self.assertRaises(FileExistsError):
                monitor.monitor(args, {}, stop, sampler=sampler)


if __name__ == "__main__":
    unittest.main()
