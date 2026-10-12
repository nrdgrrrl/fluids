"""Mocked acceptance contracts; fixture numbers are not GPU measurements."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from fluids import acceptance


GLX = "OpenGL renderer string: Intel UHD Graphics (CML GT2)\nOpenGL version string: 4.6 Mesa\n"
VULKAN = """GPU0:
    deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName = Intel UHD Graphics (CML GT2)
    driverName = Intel open-source Mesa driver
    driverInfo = Mesa 25.2.8
"""
XRANDR = """Screen 0: minimum 8 x 8, current 1920 x 1080, maximum 32767 x 32767
HDMI-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis)
   1920x1080     59.96*+  60.00
DP-1 disconnected (normal left inverted right x axis y axis)
"""
HARDWARE = acceptance.detect_hardware(GLX, VULKAN, XRANDR)


def native_fixture():
    # Artificial values solely to exercise validation/formatting without frames.
    return {"kind": "complete_native_frames", "sample_complete": True,
            "warmup_seconds": 8, "sample_seconds": 30, "fps": 2,
            "frame": {"samples": 60, "mean_ms": 500},
            "simulated_seconds": 0.25, "discarded_seconds": 0.125,
            "substeps": 4, "scalar_reads": {"counts": {"overflow": 4, "timestep": 4}, "total": 8}}


def render_fixture():
    return {"warmup_seconds": 8, "timing": {"sample_seconds": 30, "fps": 2, "mean_frame_ms": 500,
                                           "frames": 60, "mean_substeps": 0},
            "simulation_start": {"simulated_seconds": 3, "discarded_seconds": 1},
            "simulation_end": {"simulated_seconds": 3.25, "discarded_seconds": 1.125},
            "water_city_resets": [{"reset": index + 1, "particles": 5} for index in range(10)],
            "process_memory": {"sample_start_rss_bytes": None},
            "gpu_memory": "unmeasured"}


class PreflightTests(unittest.TestCase):
    def test_host_gates(self):
        acceptance.require_host((3, 11, 2), {"DISPLAY": ":0"}, ["/dev/dri/renderD128"])
        for version, environ, devices, message in (
            ((3, 10), {"DISPLAY": ":0"}, ["gpu"], "Python"),
            ((3, 12), {"DISPLAY": ":0"}, ["gpu"], "Python"),
            ((3, 11), {}, ["gpu"], "DISPLAY"),
            ((3, 11), {"DISPLAY": " "}, ["gpu"], "DISPLAY"),
            ((3, 11), {"DISPLAY": ":0"}, [], "/dev/dri"),
            ((3, 11), {"DISPLAY": ":0", "TI_ARCH": "cpu"}, ["gpu"], "TI_ARCH"),
        ):
            with self.subTest(message=message, version=version), self.assertRaisesRegex(acceptance.PreflightError, message):
                acceptance.require_host(version, environ, devices)

    def test_hardware_metadata_is_detected_and_not_a_placeholder(self):
        self.assertEqual(HARDWARE.renderer, "Intel UHD Graphics (CML GT2)")
        self.assertEqual(HARDWARE.gpu, "Intel UHD Graphics (CML GT2)")
        self.assertEqual(HARDWARE.driver, "Intel open-source Mesa driver / Mesa 25.2.8")
        self.assertEqual(HARDWARE.refresh_hz, 59.96)
        self.assertEqual(HARDWARE.displays, {"HDMI-1": 59.96})
        fallback = acceptance.detect_hardware("", VULKAN, XRANDR)
        self.assertEqual(fallback.renderer, fallback.gpu)

    def test_every_software_renderer_is_refused_in_glx_or_vulkan(self):
        for renderer in ("llvmpipe", "LAVAPIPE", "softpipe", "swrast", "PHYSICAL_DEVICE_TYPE_CPU"):
            for location in ("glx", "vulkan"):
                with self.subTest(renderer=renderer, location=location), self.assertRaisesRegex(
                        acceptance.PreflightError, "Software renderer"):
                    acceptance.detect_hardware(renderer if location == "glx" else GLX,
                                               VULKAN + renderer if location == "vulkan" else VULKAN, XRANDR)

    def test_unknown_or_ambiguous_metadata_is_refused(self):
        for vulkan, xrandr, message in (
            ("", XRANDR, "exactly one"),
            (VULKAN + VULKAN, XRANDR, "exactly one"),
            (VULKAN.replace("PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", "unknown"), XRANDR, "physical GPU type"),
            (VULKAN.split("    driverName")[0], XRANDR, "driver"),
            (VULKAN, "", "refresh rate"),
            (VULKAN, XRANDR.replace("59.96*+", "59.96+"), "refresh rate"),
            (VULKAN, XRANDR + "DP-2 connected 800x600+0+0\n   800x600 75.00*\n", "refresh rate"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(acceptance.PreflightError, message):
                acceptance.detect_hardware(GLX, vulkan, xrandr)
        acceptance.detect_hardware(GLX, VULKAN, XRANDR + "DP-2 connected 800x600+0+0\n   800x600 59.96*\n")

    def test_tool_preflight_writes_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "require_host"), patch.object(
                acceptance.subprocess, "run") as run:
            run.side_effect = [SimpleNamespace(returncode=0, stdout=value, stderr="")
                               for value in (GLX, VULKAN, XRANDR)]
            self.assertEqual(acceptance.preflight(Path(tmp)), HARDWARE)
            self.assertIn("Mesa 25.2.8", (Path(tmp) / "vulkaninfo.txt").read_text())
            self.assertEqual([call.args[0] for call in run.call_args_list],
                             [["glxinfo", "-B"], ["vulkaninfo", "--summary"], ["xrandr", "--current"]])

    def test_missing_optional_glx_is_allowed_but_required_tools_fail(self):
        for missing, expected in (("glxinfo", None), ("vulkaninfo", "vulkaninfo"), ("xrandr", "xrandr")):
            def probe(command, **kwargs):
                if command[0] == missing:
                    raise FileNotFoundError(missing)
                value = {"glxinfo": GLX, "vulkaninfo": VULKAN, "xrandr": XRANDR}[command[0]]
                return SimpleNamespace(returncode=0, stdout=value, stderr="")

            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance, "require_host"), patch.object(acceptance.subprocess, "run", side_effect=probe):
                if expected:
                    with self.assertRaisesRegex(acceptance.PreflightError, expected):
                        acceptance.preflight(Path(tmp))
                else:
                    self.assertEqual(acceptance.preflight(Path(tmp)).gpu, HARDWARE.gpu)

    def test_no_gpu_stops_before_system_tools(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DISPLAY": ":0"}), patch.object(
                acceptance.Path, "glob", return_value=[]), patch.object(acceptance.subprocess, "run") as run:
            with self.assertRaisesRegex(acceptance.PreflightError, "/dev/dri"):
                acceptance.preflight(Path(tmp))
            run.assert_not_called()


class AvailabilityTests(unittest.TestCase):
    def test_successful_help_detects_exact_budget_flag_and_archives_evidence(self):
        for help_text, available in (("usage: benchmark [--physics-budget-ms MS]\n", True),
                                     ("usage: benchmark [--steps STEPS]\n", False),
                                     ("usage: benchmark [--physics-budget-ms-extra MS]\n", False)):
            with self.subTest(help_text=help_text), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance.subprocess, "run", return_value=SimpleNamespace(
                        returncode=0, stdout=help_text, stderr="mock startup")) as run:
                reason = acceptance.probe_issue_39(Path(tmp))
                self.assertEqual(reason is None, available)
                if not available:
                    self.assertIn("does not advertise --physics-budget-ms", reason)
                run.assert_called_once_with([acceptance.sys.executable, "-m", "fluids.benchmark", "--help"],
                                            capture_output=True, text=True, timeout=30)
                self.assertEqual((Path(tmp) / "benchmark-help.stdout.txt").read_text(), help_text)
                self.assertEqual((Path(tmp) / "benchmark-help.stderr.txt").read_text(), "mock startup")

    def test_failed_help_is_not_treated_as_an_absent_or_supported_flag(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance.subprocess, "run", return_value=
                SimpleNamespace(returncode=1, stdout="--physics-budget-ms", stderr="missing dependency")):
            reason = acceptance.probe_issue_39(Path(tmp))
            self.assertIn("exited 1", reason)
            self.assertNotIn("does not advertise", reason)
            self.assertEqual((Path(tmp) / "benchmark-help.stderr.txt").read_text(), "missing dependency")

    def test_missing_process_and_timeout_have_truthful_skip_reasons(self):
        for error in (FileNotFoundError("mock missing interpreter"), subprocess.TimeoutExpired("help", 30)):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance.subprocess, "run", side_effect=error):
                reason = acceptance.probe_issue_39(Path(tmp))
                self.assertIn("Could not probe", reason)
                self.assertIn(str(error), reason)
                self.assertIn(str(error), (Path(tmp) / "benchmark-help.stderr.txt").read_text())


class CaseTests(unittest.TestCase):
    def test_all_documented_cases_and_sampling_are_present(self):
        cases = acceptance.build_cases()
        self.assertEqual(len(cases), 45)
        self.assertEqual(len({case.name for case in cases}), 45)
        self.assertEqual(sum(case.graphical for case in cases), 31)
        self.assertEqual(sum(case.capture for case in cases), 6)
        for case in cases:
            flags = case.flags
            if case.capture:
                for flag, value in (("--warmup", "8"), ("--seconds", "30"), ("--resets", "10")):
                    self.assertEqual(flags[flags.index(flag) + 1], value)
            elif case.graphical:
                for flag, value in (("--mode", "native"), ("--warmup-seconds", "8"), ("--sample-seconds", "30")):
                    self.assertEqual(flags[flags.index(flag) + 1], value)
            else:
                self.assertEqual(flags[flags.index("--backend") + 1], "vulkan")
                self.assertEqual(flags[flags.index("--steps") + 1], "40")
        names = {case.name for case in cases}
        self.assertIn("36-legacy-idle", names)
        self.assertIn("36-fixed-wave", names)
        self.assertIn("36-quality-paused", names)
        self.assertIn("38-duplicate-medium-idle", names)
        self.assertIn("38-cached-low-wave", names)
        self.assertIn("37-stock-paused", names)
        self.assertIn("37-device-active", names)

    def test_selector_filters_exactly_one_issue(self):
        for issue, size in ((36, 15), (37, 6), (38, 12), (39, 12)):
            cases = acceptance.build_cases(issue)
            self.assertEqual(len(cases), size)
            self.assertTrue(all(case.issue == issue for case in cases))

    def test_issue_39_budget_pairs_cover_native_scenarios_and_stage_diagnostics(self):
        pairs = {}
        native_scenarios = set()
        stage_scenarios = set()
        for case in acceptance.build_cases(39):
            self.assertEqual(case.module, "fluids.benchmark")
            self.assertFalse(case.capture)
            flags = list(case.flags)
            self.assertEqual(flags[flags.index("--backend") + 1], "vulkan")
            index = flags.index("--physics-budget-ms")
            budget = flags[index + 1]
            del flags[index:index + 2]
            pairs.setdefault(tuple(flags), []).append(budget)
            if case.graphical:
                values = dict(zip(flags[::2], flags[1::2]))
                self.assertEqual(values["--mode"], "native")
                self.assertEqual(values["--warmup-seconds"], "8")
                self.assertEqual(values["--sample-seconds"], "30")
                native_scenarios.add((values["--scenario"], values["--quality"]))
            else:
                self.assertEqual(flags[flags.index("--steps") + 1], "40")
                self.assertEqual(flags[flags.index("--warmup-steps") + 1], "8")
                if "--wave" in flags:
                    self.assertEqual(flags[flags.index("--presteps") + 1], "120")
                else:
                    self.assertNotIn("--presteps", flags)
                stage_scenarios.add("--wave" in flags)
        self.assertEqual(native_scenarios, {("idle", "medium"), ("wave", "medium"),
                                            ("paused", "medium"), ("wave", "low")})
        self.assertEqual(stage_scenarios, {False, True})
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(sorted(budgets) == ["0", "12"] for budgets in pairs.values()))

    def test_unavailable_issue_39_does_not_change_other_cases(self):
        cases = acceptance.build_cases(issue_39_available=False)
        self.assertEqual(cases, [case for case in acceptance.build_cases() if case.issue != 39])
        self.assertEqual(acceptance.build_cases(39, issue_39_available=False), [])

    def test_commands_use_current_python_detected_labels_and_unique_paths(self):
        for case in acceptance.build_cases():
            command = acceptance.case_command(case, HARDWARE, Path("/results"))
            self.assertEqual(command[:3], [acceptance.sys.executable, "-m", case.module])
            self.assertIn(HARDWARE.gpu, command)
            self.assertIn(HARDWARE.driver, command)
            self.assertEqual(command[command.index("--refresh-hz") + 1], "59.96")
            if case.capture:
                self.assertIn(f"/results/{case.name}.png", command)
                self.assertIn(f"/results/{case.name}.json", command)


class ExecutionTests(unittest.TestCase):
    def test_report_extraction_handles_startup_banners(self):
        fixture = native_fixture()
        self.assertEqual(acceptance.read_report("[Taichi] version\n[Taichi] Starting on arch=vulkan\n" +
                                                json.dumps(fixture, indent=2)), fixture)
        with self.assertRaisesRegex(ValueError, "JSON report"):
            acceptance.read_report("startup failed\n")

    def test_incomplete_or_invalid_samples_never_become_measurements(self):
        case = acceptance.build_cases(36)[0]
        acceptance.validate_report(case, native_fixture())
        for key, value in (("sample_complete", False), ("sample_seconds", 29), ("warmup_seconds", 7.9),
                           ("fps", 0), ("fps", float("nan")), ("discarded_seconds", float("inf"))):
            report = dict(native_fixture(), **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                acceptance.validate_report(case, report)
        render = acceptance.build_cases(37)[0]
        acceptance.validate_report(render, render_fixture())
        report = render_fixture()
        report["water_city_resets"].pop()
        with self.assertRaisesRegex(ValueError, "reset"):
            acceptance.validate_report(render, report)

    def test_fresh_process_for_each_case_and_capture_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def fake_harness(command, stdout, stderr, **kwargs):
                report = render_fixture() if "--capture" in command else native_fixture()
                stdout.write("[Taichi] startup\n" + json.dumps(report))
                if "--capture" in command:
                    Path(command[command.index("--capture") + 1]).write_bytes(b"mock capture, not rendered pixels")
                return SimpleNamespace(returncode=0)

            cases = [acceptance.build_cases(36)[0], acceptance.build_cases(37)[0]]
            with patch.object(acceptance.subprocess, "run", side_effect=fake_harness) as run:
                for case in cases:
                    result = acceptance.execute_case(case, HARDWARE, directory)
                    self.assertEqual(result["status"], "measured")
                    report = json.loads((directory / f"{case.name}.json").read_text())
                    self.assertEqual(report["acceptance_hardware"]["renderer"], HARDWARE.renderer)
                self.assertEqual(run.call_count, 2)

            def no_capture(command, stdout, stderr, **kwargs):
                stdout.write(json.dumps(render_fixture()))
                return SimpleNamespace(returncode=0)

            with patch.object(acceptance.subprocess, "run", side_effect=no_capture):
                result = acceptance.execute_case(acceptance.build_cases(37)[1], HARDWARE, directory)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("report", result)
            self.assertIn("PNG", result["reason"])

    def test_failure_timeout_and_partial_sample_preserve_logs_without_metrics(self):
        case = acceptance.build_cases(36)[0]
        for outcome in ("exit", "timeout", "partial"):
            def harness(command, stdout, stderr, **kwargs):
                stderr.write("mock failure diagnostic")
                stdout.write(json.dumps(dict(native_fixture(), sample_complete=False)))
                if outcome == "timeout":
                    raise subprocess.TimeoutExpired(command, 1800)
                return SimpleNamespace(returncode=1 if outcome == "exit" else 0)

            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance.subprocess, "run", side_effect=harness):
                result = acceptance.execute_case(case, HARDWARE, Path(tmp))
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("report", result)
                self.assertTrue((Path(tmp) / f"{case.name}.stderr.txt").exists())
                self.assertFalse((Path(tmp) / f"{case.name}.json").exists())


class SummaryAndArchiveTests(unittest.TestCase):
    def setUp(self):
        probe = patch.object(acceptance, "probe_issue_39", return_value=None)
        self.probe = probe.start()
        self.addCleanup(probe.stop)

    def test_summary_separates_graphical_diagnostic_correctness_and_failures(self):
        native = native_fixture()
        native.update(os="mock test host", physics_settings={"quality": "medium", "water_depth": 2.0})
        rows = [
            {"name": "mock-native", "graphical": True, "status": "measured", "report": native},
            {"name": "mock-render", "graphical": True, "status": "measured", "report": render_fixture()},
            {"name": "mock-stage", "graphical": False, "status": "diagnostic",
             "report": {"simulated_seconds": 0.1, "scalar_reads": {"total": 8}}},
            {"name": "failed", "graphical": True, "status": "failed", "reason": "window closed"},
            {"name": "skipped", "graphical": True, "status": "skipped", "reason": "prerequisite"},
        ]
        summary = acceptance.format_summary(rows, HARDWARE, [{"name": "CPU test", "status": "passed"}])
        self.assertIn("NON-PERFORMANCE", summary)
        self.assertIn("Host: os=mock test host", summary)
        self.assertIn("quality=medium", summary)
        self.assertIn("not frame/FPS measurements", summary)
        self.assertIn("mock-native: measured graphically; FPS=2; frame_ms=500; simulated_s=0.25; discarded_s=0.125", summary)
        self.assertIn("mock-render: measured graphically; FPS=2; frame_ms=500; simulated_s=0.25; discarded_s=0.125", summary)
        self.assertIn("scalar_reads.counts.overflow=4", summary)
        self.assertIn("water_city_resets.reset=[1,2,3,4,5,6,7,8,9,10]", summary)
        self.assertIn("water_city_resets.particles[all 10]=5", summary)
        self.assertIn("mock-stage: diagnostic; FPS/frame=unmeasured", summary)
        self.assertIn("failed: failed — unmeasured", summary)
        self.assertIn("skipped: skipped — unexecuted/unmeasured", summary)

    def test_archive_contains_reports_captures_logs_manifest_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "results"
            directory.mkdir()
            for filename in ("case.json", "case.png", "case.stdout.txt", "vulkaninfo.txt"):
                (directory / filename).write_text("mock artifact")
            archive = acceptance.assemble_archive(directory, {"cases": []}, "mock summary\n")
            self.assertEqual(archive, directory.with_suffix(".zip"))
            with zipfile.ZipFile(archive) as zipped:
                for filename in ("case.json", "case.png", "case.stdout.txt", "vulkaninfo.txt", "manifest.json", "summary.txt"):
                    self.assertIn(f"results/{filename}", zipped.namelist())
                self.assertEqual(json.loads(zipped.read("results/manifest.json")), {"cases": []})
                self.assertEqual(zipped.read("results/summary.txt"), b"mock summary\n")

    def test_main_preflight_refusal_archives_unmeasured_cases_and_never_executes(self):
        with (tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", side_effect=
                acceptance.PreflightError("Software renderer detected")), patch.object(acceptance, "execute_case") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout):
            directory = Path(tmp) / "run"
            code = acceptance.main(["--results", str(directory), "--only", "37"])
            self.assertEqual(code, 1)
            run.assert_not_called()
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(len(manifest["cases"]), 6)
            self.assertTrue(all(row["status"] == "skipped" for row in manifest["cases"]))
            self.assertNotIn("FPS=", stdout.getvalue())
            self.assertIn("unexecuted/unmeasured", stdout.getvalue())
            self.assertIn(f"Results archive: {directory}.zip", stdout.getvalue())

    def test_main_continues_after_case_failure_and_reports_it(self):
        cases = acceptance.build_cases(37)[:2]
        failure = {"name": cases[0].name, "graphical": True, "status": "failed", "reason": "mock fail"}
        success = {"name": cases[1].name, "graphical": True, "status": "measured", "report": render_fixture()}
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", return_value=HARDWARE), patch.object(
                acceptance, "build_cases", return_value=cases), patch.object(acceptance, "execute_case",
                side_effect=[failure, success]) as run, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory)]), 1)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(directory.with_suffix(".zip").exists())

    def test_optional_checks_are_non_performance_and_failure_blocks_cases(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", return_value=HARDWARE), patch.object(
                acceptance, "run_checks", return_value=[{"name": "CPU suite", "status": "failed"}]), patch.object(
                acceptance, "execute_case") as run, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory), "--with-checks"]), 1)
            run.assert_not_called()

    def test_only_39_runs_all_available_cases_and_separates_stage_diagnostics(self):
        def measured(case, hardware, directory):
            return {"name": case.name, "issue": case.issue, "graphical": case.graphical,
                    "status": "measured" if case.graphical else "diagnostic",
                    "report": native_fixture() if case.graphical else
                    {"kind": "synchronized_evolving_stages", "simulated_seconds": 0.1}}

        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", return_value=HARDWARE), patch.object(
                acceptance, "execute_case", side_effect=measured) as run, contextlib.redirect_stdout(io.StringIO()) as stdout:
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory), "--only", "39"]), 0)
            self.assertEqual([call.args[0] for call in run.call_args_list], acceptance.build_cases(39))
            summary = stdout.getvalue()
            self.assertNotIn("Issue #39: skipped", summary)
            graphical, stages = summary.split("SYNCHRONIZED Vulkan stage diagnostics")
            self.assertIn("39-budget-12-medium-wave: measured graphically", graphical)
            self.assertIn("39-budget-0-low-wave: measured graphically", graphical)
            self.assertIn("39-budget-12-stages-idle: diagnostic; FPS/frame=unmeasured", stages)
            self.assertIn("39-budget-0-stages-wave: diagnostic; FPS/frame=unmeasured", stages)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["issue_39"], {"status": "available"})
            self.assertEqual(len(manifest["cases"]), 12)

    def test_unavailable_issue_39_is_skipped_with_probe_reason(self):
        reason = "fluids.benchmark --help does not advertise --physics-budget-ms in this checkout."
        self.probe.return_value = reason

        def measured(case, hardware, directory):
            return {"name": case.name, "issue": case.issue, "graphical": case.graphical,
                    "status": "measured" if case.graphical else "diagnostic",
                    "report": render_fixture() if case.capture else native_fixture() if case.graphical else {}}

        for only, expected, size in (([], 0, 33), (["--only", "39"], 1, 0)):
            with self.subTest(only=only), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance, "preflight", return_value=HARDWARE), patch.object(acceptance, "execute_case",
                    side_effect=measured) as run, contextlib.redirect_stdout(io.StringIO()) as stdout:
                directory = Path(tmp) / "run"
                self.assertEqual(acceptance.main(["--results", str(directory), *only]), expected)
                self.assertEqual(run.call_count, size)
                self.assertIn(f"Issue #39: skipped — {reason}", stdout.getvalue())
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["issue_39"], {"status": "skipped", "reason": reason})
                self.assertTrue(all(row["issue"] != 39 for row in manifest["cases"]))
                self.assertTrue(directory.with_suffix(".zip").exists())

    def test_only_39_preflight_refusal_archives_every_case_without_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", side_effect=
                acceptance.PreflightError("DISPLAY is unset")), patch.object(acceptance, "execute_case") as run, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory), "--only", "39"]), 1)
            run.assert_not_called()
            rows = json.loads((directory / "manifest.json").read_text())["cases"]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["status"] == "skipped" and "report" not in row for row in rows))
            self.assertNotIn("FPS=", stdout.getvalue())
            self.assertIn("unexecuted/unmeasured", stdout.getvalue())
            self.assertTrue(directory.with_suffix(".zip").exists())

    def test_other_issue_selector_does_not_probe_issue_39(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", side_effect=
                acceptance.PreflightError("DISPLAY is unset")), contextlib.redirect_stdout(io.StringIO()):
            acceptance.main(["--results", str(Path(tmp) / "run"), "--only", "37"])
            self.probe.assert_not_called()

    def test_existing_results_or_archive_are_never_overwritten(self):
        for artifact in ("run", "run.zip"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(
                    io.StringIO()), patch.object(acceptance, "preflight") as preflight:
                path = Path(tmp) / artifact
                path.write_text("previous results")
                with self.assertRaises(SystemExit) as exit_status:
                    acceptance.main(["--results", str(Path(tmp) / "run")])
                self.assertEqual(exit_status.exception.code, 2)
                self.assertEqual(path.read_text(), "previous results")
                preflight.assert_not_called()

    def test_check_commands_force_cpu_and_disable_optional_native_test(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance.subprocess, "run", return_value=
                SimpleNamespace(returncode=0)) as run, contextlib.redirect_stdout(io.StringIO()), patch.dict(
                os.environ, {"TI_ARCH": "vulkan", "FLUIDS_TEST_NATIVE": "1", "FLUIDS_TEST_PARTICLE_ARCH": "vulkan"}):
            checks = acceptance.run_checks(Path(tmp))
            self.assertEqual(len(checks), 3)
            self.assertTrue(all(check["status"] == "passed" for check in checks))
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"]["FLUIDS_TEST_PARTICLE_ARCH"], "cpu")
                self.assertEqual(call.kwargs["env"]["FLUIDS_TEST_NATIVE"], "0")
                self.assertNotIn("TI_ARCH", call.kwargs["env"])
            self.assertEqual(run.call_args_list[-1].args[0][-2:], ["-m", "fluids.smoke"])

    def test_interrupt_archives_failed_active_and_skipped_remaining_cases(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", return_value=HARDWARE), patch.object(
                acceptance, "execute_case", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory), "--only", "37"]), 1)
            rows = json.loads((directory / "manifest.json").read_text())["cases"]
            self.assertEqual(rows[0]["status"], "failed")
            self.assertTrue(all(row["status"] == "skipped" for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
