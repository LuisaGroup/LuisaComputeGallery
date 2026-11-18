#!/usr/bin/env python3
"""Stdlib unit tests for nsight_profile.py; no NVIDIA GPU is required."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nsight_profile as np


class CommandConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nsys = Path(r"C:\Nsight\nsys.exe")
        self.ncu = Path(r"C:\Nsight\ncu.exe")
        self.app = [r"C:\gallery\example.exe", "cuda", "--headless", "--frames", "1"]

    def profile_args(self, *extra: str):
        return np.build_parser().parse_args(["profile", "example_path_tracing", *extra])

    def test_nsys_cuda_defaults(self) -> None:
        args = self.profile_args()
        argv = np.nsys_profile_argv(self.nsys, self.app, Path("system"), args, "cuda")
        self.assertIn("--trace=cuda,nvtx", argv)
        self.assertIn("--sample=none", argv)
        self.assertIn("--cpuctxsw=none", argv)
        self.assertIn("--duration=0", argv)
        self.assertIn("--show-output=true", argv)
        self.assertEqual(argv[-len(self.app):], self.app)
        self.assertLess(argv.index("--output=system"), argv.index(self.app[0]))

    def test_nsys_backend_specific_providers(self) -> None:
        args = self.profile_args("--duration=2.5", "--kill=false", "--force",
                                 "--resolve-symbols", "--gpu-metrics")
        dx = np.nsys_profile_argv(self.nsys, self.app, Path("dx"), args, "dx")
        vk = np.nsys_profile_argv(self.nsys, self.app, Path("vk"), args, "vk")
        self.assertIn("--trace=dx12,dx12-annotations,nvtx", dx)
        self.assertIn("--trace=vulkan,vulkan-annotations,nvtx", vk)
        self.assertIn("--duration=2.5", dx)
        self.assertIn("--kill=false", dx)
        self.assertIn("--force-overwrite=true", dx)
        self.assertIn("--resolve-symbols=true", dx)
        self.assertIn("--gpu-metrics-devices=all", dx)

    def test_nsys_custom_trace_and_advanced_args(self) -> None:
        args = self.profile_args("--trace=cuda,nvtx,cublas", "--nsys-arg=--stats=true")
        argv = np.nsys_profile_argv(self.nsys, self.app, Path("out"), args, "cuda")
        self.assertIn("--trace=cuda,nvtx,cublas", argv)
        self.assertEqual(argv[-len(self.app) - 1], "--stats=true")

    def test_nsys_stats_uses_terminal_output_and_sqlite(self) -> None:
        argv = np.nsys_stats_argv(self.nsys, Path("a.nsys-rep"), Path("a.sqlite"),
                                  None, "cuda", True)
        self.assertEqual(argv[:2], [str(self.nsys), "stats"])
        self.assertIn("--format=table", argv)
        self.assertIn("--output=-", argv)
        self.assertIn("--sqlite=a.sqlite", argv)
        self.assertIn("--force-export=true", argv)
        self.assertIn("--force-overwrite=true", argv)
        self.assertEqual(argv[-1], "a.nsys-rep")

    def test_ncu_bounds_replay_and_import(self) -> None:
        args = self.profile_args("--mode=ncu", "--ncu-set=detailed",
                                 "--launch-count=3", "--launch-skip=7",
                                 "--kernel-name=regex:render.*", "--force")
        argv = np.ncu_profile_argv(self.ncu, self.app, Path("kernel"), args)
        self.assertEqual(argv[argv.index("--set") + 1], "detailed")
        self.assertEqual(argv[argv.index("--launch-count") + 1], "3")
        self.assertEqual(argv[argv.index("--launch-skip") + 1], "7")
        self.assertEqual(argv[argv.index("--kernel-name") + 1], "regex:render.*")
        self.assertIn("--force-overwrite", argv)
        report = np.ncu_report_argv(self.ncu, Path("kernel.ncu-rep"), "raw", True)
        self.assertEqual(report, [str(self.ncu), "--import", "kernel.ncu-rep",
                                  "--page", "raw", "--csv"])

    def test_ncu_all_launches_omits_count(self) -> None:
        args = self.profile_args("--mode=ncu", "--all-launches")
        argv = np.ncu_profile_argv(self.ncu, self.app, Path("kernel"), args)
        self.assertNotIn("--launch-count", argv)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.targets = np.load_targets()

    def test_manifest_contains_only_desktop_binaries(self) -> None:
        self.assertTrue(self.targets)
        self.assertTrue(all(target.get("kind", "binary") == "binary" for target in self.targets))
        self.assertTrue(all(target["group"] != "ios" for target in self.targets))
        self.assertIn("example_path_tracing", {target["name"] for target in self.targets})

    def test_application_arguments_follow_manifest_backend_style(self) -> None:
        target = {"name": "custom", "backend_style": "flag-equals",
                  "headless": ["--model", "$ROOT/data/model"]}
        argv = np.application_argv(Path("demo.exe"), target, "cuda", ["--extra"], True)
        self.assertEqual(argv[:4], ["demo.exe", "--backend=cuda", "--headless", "--model"])
        self.assertEqual(Path(argv[4]), np.ROOT / "data/model")
        self.assertEqual(argv[5], "--extra")
        without_defaults = np.application_argv(Path("demo.exe"), target, "vk", [], False)
        self.assertEqual(without_defaults, ["demo.exe", "--backend=vk", "--headless"])

    def test_target_selection_and_unknown_name(self) -> None:
        selected = np.select_targets(self.targets, [], "path_tracing_camera")
        self.assertEqual([target["name"] for target in selected],
                         ["example_path_tracing_camera"])
        with self.assertRaisesRegex(np.ProfileError, "unknown gallery target"):
            np.select_targets(self.targets, ["example_path_tracing_camra"], None)
        with self.assertRaisesRegex(np.ProfileError, "select at least one"):
            np.select_targets(self.targets, [], None)

    def test_backend_specific_skip_reason(self) -> None:
        target = {"headless_skip": {"cuda": "unsupported"}}
        self.assertEqual(np.skip_reason(target, "cuda"), "unsupported")
        self.assertIsNone(np.skip_reason(target, "dx"))


class DiscoveryAndOutputTests(unittest.TestCase):
    def test_explicit_installation_root_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Nsight Systems 2099.1.2"
            executable = root / "target-windows-x64" / "nsys.exe"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"test")
            self.assertEqual(np.discover_tool("nsys", str(root)), executable.resolve())

    def test_missing_explicit_tool_is_an_error(self) -> None:
        with self.assertRaisesRegex(np.ProfileError, "explicit path"):
            np.discover_tool("ncu", r"C:\does-not-exist\ncu.exe")
        self.assertIsNone(np.discover_tool("ncu", r"C:\does-not-exist\ncu.exe",
                                          required=False))

    def test_dry_run_root_does_not_create_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            root = np._prepare_run_root(str(output), "dry_run-1", False, True)
            self.assertEqual(root, output.resolve() / "dry_run-1")
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(np.ProfileError, "run-name"):
            np._prepare_run_root("unused", "../escape", False, True)

    def test_real_run_root_refuses_accidental_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = np._prepare_run_root(temporary, "same", False, False)
            self.assertTrue(first.is_dir())
            with self.assertRaisesRegex(np.ProfileError, "already exists"):
                np._prepare_run_root(temporary, "same", False, False)

    def test_failed_target_report_is_salvaged_but_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "system.nsys-rep"
            failed = {"status": "fail"}
            self.assertFalse(np._should_generate_report(failed, report, False, False))
            report.write_bytes(b"collected before target failure")
            self.assertTrue(np._should_generate_report(failed, report, False, False))
            self.assertFalse(np._should_generate_report(failed, report, True, False))
            self.assertTrue(np._should_generate_report({"status": "planned"}, report,
                                                       False, True))

    def test_run_logged_preserves_output_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cwd = Path(temporary)
            ok_log = cwd / "ok.log"
            ok = np.run_logged([sys.executable, "-c", "print('profiler-output')"],
                               ok_log, cwd, 20, False)
            self.assertEqual(ok["status"], "ok")
            self.assertEqual(ok["returncode"], 0)
            self.assertIn("profiler-output", ok_log.read_text(encoding="utf-8"))
            fail = np.run_logged([sys.executable, "-c", "raise SystemExit(7)"],
                                 cwd / "fail.log", cwd, 20, False)
            self.assertEqual(fail["status"], "fail")
            self.assertEqual(fail["returncode"], 7)


class WorkflowTests(unittest.TestCase):
    def test_combined_profile_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifacts"
            with contextlib.redirect_stdout(io.StringIO()):
                result = np.main(["profile", "example_path_tracing", "--mode", "both",
                                  "--dry-run", "--output-dir", str(output),
                                  "--run-name", "check"])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())

    def test_ncu_rejects_non_cuda_backend(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            result = np.main(["profile", "example_path_tracing", "--mode", "ncu",
                              "--backend", "dx", "--dry-run"])
        self.assertEqual(result, 2)

    def test_offline_report_dry_run_for_both_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "system.nsys-rep").write_bytes(b"placeholder")
            (root / "kernel.ncu-rep").write_bytes(b"placeholder")
            with contextlib.redirect_stdout(io.StringIO()):
                result = np.main(["report", str(root), "--dry-run"])
            self.assertEqual(result, 0)
            self.assertFalse((root / "system.stats.txt").exists())
            self.assertFalse((root / "kernel.details.txt").exists())

    def test_collection_failure_marks_summary_and_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe_dir = root / "bin"
            exe_dir.mkdir()
            (exe_dir / "example_path_tracing.exe").write_bytes(b"placeholder")
            failed = {"status": "fail", "returncode": 1, "seconds": 0.0,
                      "argv": [], "command": "failed", "log": "log"}
            with mock.patch.object(np, "discover_tool", return_value=Path("nsys.exe")), \
                    mock.patch.object(np, "run_logged", return_value=failed.copy()), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = np.main(["profile", "example_path_tracing", "--output-dir",
                                  str(root / "out"), "--run-name", "failed",
                                  "--exe-dir", str(exe_dir)])
            self.assertEqual(result, 1)
            summary = json.loads((root / "out" / "failed" / "summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual(summary["results"][0]["status"], "fail")
            self.assertEqual(summary["results"][0]["operations"][0]["kind"], "nsys-profile")


if __name__ == "__main__":
    unittest.main()
