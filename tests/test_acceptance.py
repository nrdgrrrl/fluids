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
from unittest.mock import Mock, patch
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


def render_fixture(resets=10):
    return {"sample_complete": True, "panel_fps": 2, "warmup_seconds": 8, "timing": {"sample_seconds": 30, "fps": 2, "mean_frame_ms": 500,
                                           "frames": 60, "mean_substeps": 0},
            "simulation_start": {"simulated_seconds": 3, "discarded_seconds": 1},
            "simulation_end": {"simulated_seconds": 3.25, "discarded_seconds": 1.125},
            "water_city_resets": [{"reset": index + 1, "particles": 5} for index in range(resets)],
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
    def test_default_is_exactly_two_visual_application_runs(self):
        cases = acceptance.build_cases()
        self.assertEqual([case.name for case in cases], ["baseline", "current"])
        self.assertEqual(cases[0].flags, cases[1].flags)
        self.assertTrue(all(case.graphical and case.capture for case in cases))
        self.assertTrue(all(case.module == "fluids.acceptance" for case in cases))
        for case in cases:
            self.assertEqual(acceptance.graphical_window(case), (2, 25))
            self.assertEqual(acceptance.reset_count(case), 0)

    def test_all_documented_cases_and_sampling_are_present(self):
        cases = acceptance.build_cases(full=True)
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
        for full, sizes in ((False, ((36, 3), (37, 3), (38, 2), (39, 2))),
                            (True, ((36, 15), (37, 6), (38, 12), (39, 12)))):
            for issue, size in sizes:
                with self.subTest(full=full, issue=issue):
                    cases = acceptance.build_cases(issue, full=full)
                    self.assertEqual(len(cases), size)
                    self.assertTrue(all(case.issue == issue for case in cases))

    def test_timing_overrides_reach_both_harnesses_but_leave_diagnostics_unchanged(self):
        for full in (False, True):
            defaults = {case.name: case for case in acceptance.build_cases(full=full)}
            cases = acceptance.build_cases(full=full, warmup_seconds=0.5, sample_seconds=2.5)
            for case in cases:
                if case.graphical:
                    self.assertEqual(acceptance.graphical_window(case), (0.5, 2.5))
                    command = acceptance.case_command(case, HARDWARE, Path("/results"))
                    self.assertIn("0.5", command)
                    self.assertIn("2.5", command)
                else:
                    self.assertEqual(case, defaults[case.name])

    def test_plan_counts_actual_windows_diagnostics_and_resets(self):
        default = acceptance.plan_cases(acceptance.build_cases())
        self.assertEqual(default, {"cases": 2, "graphical": 2, "diagnostics": 0, "resets": 0,
                                  "window_seconds": 54, "estimated_min_seconds": 94,
                                  "estimated_max_seconds": 174})
        full = acceptance.plan_cases(acceptance.build_cases(full=True))
        self.assertEqual((full["cases"], full["graphical"], full["diagnostics"], full["resets"]),
                         (45, 31, 14, 60))
        self.assertEqual(full["window_seconds"], 31 * 38)
        shorter = acceptance.plan_cases(acceptance.build_cases(39, sample_seconds=5, warmup_seconds=1))
        self.assertEqual(shorter["cases"], 2)
        self.assertEqual(shorter["window_seconds"], 12)
        self.assertEqual(shorter["estimated_min_seconds"], 52)
        self.assertEqual(acceptance.plan_cases([])["estimated_max_seconds"], 0)

    def test_issue_39_budget_pairs_cover_native_scenarios_and_stage_diagnostics(self):
        pairs = {}
        native_scenarios = set()
        stage_scenarios = set()
        for case in acceptance.build_cases(39, full=True):
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
        for full in (False, True):
            cases = acceptance.build_cases(issue_39_available=False, full=full)
            self.assertEqual(cases, [case for case in acceptance.build_cases(full=full) if case.issue != 39])
            self.assertEqual(acceptance.build_cases(39, issue_39_available=False, full=full), [])

    def test_commands_use_current_python_detected_labels_and_unique_paths(self):
        for case in acceptance.build_cases(full=True):
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
        render = acceptance.build_cases(37, full=True)[0]
        acceptance.validate_report(render, render_fixture())
        report = render_fixture()
        report["water_city_resets"].pop()
        with self.assertRaisesRegex(ValueError, "reset"):
            acceptance.validate_report(render, report)

    def test_quick_samples_validate_requested_window_and_default_has_no_reset_requirement(self):
        for full in (False, True):
            for issue in (36, 37):
                with self.subTest(full=full, issue=issue):
                    case = acceptance.build_cases(issue, full=full, warmup_seconds=1, sample_seconds=3)[0]
                    report = render_fixture(10 if full else 0) if case.capture else native_fixture()
                    timing = report["timing"] if case.capture else report
                    report["warmup_seconds"] = 1
                    timing["sample_seconds"] = 3
                    acceptance.validate_report(case, report)
                    timing["sample_seconds"] = 2.9
                    with self.assertRaisesRegex(ValueError, "3s sample"):
                        acceptance.validate_report(case, report)
                    timing["sample_seconds"] = 3
                    report["warmup_seconds"] = 0.9
                    with self.assertRaisesRegex(ValueError, "1s warm-up"):
                        acceptance.validate_report(case, report)

    def test_fresh_process_for_each_case_and_capture_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def fake_harness(command, stdout, stderr, **kwargs):
                report = (render_fixture(int(command[command.index("--resets") + 1]))
                          if "--capture" in command else native_fixture())
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
                stdout.write(json.dumps(render_fixture(0)))
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


class BaselineTests(unittest.TestCase):
    def test_setup_failures_are_skips_with_logs(self):
        for outcomes, message in (([SimpleNamespace(returncode=1), SimpleNamespace(returncode=1)], "fetch"),
                                  ([SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)], "create worktree"),
                                  ([FileNotFoundError("git unavailable")], "git unavailable"),
                                  ([subprocess.TimeoutExpired("git", 5)], "timed out")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance.subprocess, "run", side_effect=outcomes) as run:
                with acceptance.baseline_worktree(Path(tmp)) as (checkout, reason):
                    self.assertIsNone(checkout)
                    self.assertIn("1c3132b", reason)
                    self.assertIn(message, reason)
                    self.assertIn("Continuing with the current build", reason)
                    self.assertIn("unexecuted/unmeasured", reason)
                self.assertTrue((Path(tmp) / "baseline-worktree.txt").exists())
                self.assertFalse(any("remove" in call.args[0] for call in run.call_args_list))
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs["timeout"], 5)
                    self.assertEqual(call.kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")

    def test_temporary_directory_failure_still_yields_skip(self):
        with patch.object(acceptance.tempfile, "TemporaryDirectory", side_effect=OSError("unwritable temp")):
            with acceptance.baseline_worktree(Path("/unused")) as (checkout, reason):
                self.assertIsNone(checkout)
                self.assertIn("unwritable temp", reason)

    def test_detached_checkout_is_cleaned_up_even_on_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance.subprocess, "run", return_value=
                SimpleNamespace(returncode=0)) as run:
            with self.assertRaises(KeyboardInterrupt):
                with acceptance.baseline_worktree(Path(tmp)) as (checkout, reason):
                    self.assertIsNone(reason)
                    self.assertNotEqual(checkout.parent, Path(tmp))
                    raise KeyboardInterrupt()
            commands = [call.args[0][3:] for call in run.call_args_list]
            self.assertEqual(commands, [["rev-parse", "--verify", "1c3132b^{commit}"],
                                       ["worktree", "add", "--detach", str(checkout), "1c3132b"],
                                       ["worktree", "remove", "--force", str(checkout)]])

    def test_fetch_is_attempted_when_commit_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance.subprocess, "run", side_effect=
                [SimpleNamespace(returncode=code) for code in (1, 0, 0, 0)]) as run:
            with acceptance.baseline_worktree(Path(tmp)) as (checkout, reason):
                self.assertIsNotNone(checkout)
                self.assertIsNone(reason)
            self.assertEqual(run.call_args_list[1].args[0][3:], ["fetch", "--no-tags", "origin", "1c3132b"])

    def test_failed_baseline_setup_runs_only_current_and_exits_successfully(self):
        for failure in ([SimpleNamespace(returncode=1), SimpleNamespace(returncode=1)],
                        [SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)],
                        [FileNotFoundError("no git")], [subprocess.TimeoutExpired("git", 5)]):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance, "preflight", return_value=HARDWARE), patch.object(
                    acceptance.subprocess, "run", side_effect=failure), patch.object(
                    acceptance, "probe_issue_39") as probe, patch.object(
                    acceptance, "execute_case") as execute, contextlib.redirect_stdout(io.StringIO()) as output:
                report = render_fixture(0)
                report["warmup_seconds"] = 2
                report["timing"]["sample_seconds"] = 25
                directory = Path(tmp) / "results"
                execute.return_value = {"name": "current", "issue": 46, "graphical": True, "status": "measured",
                                        "report": report, "capture": str(directory / "current.png")}
                self.assertEqual(acceptance.main(["--results", str(directory)]), 0)
                probe.assert_not_called()
                execute.assert_called_once()
                self.assertEqual(execute.call_args.args[0].name, "current")
                self.assertEqual(execute.call_args.kwargs["checkout"], Path(acceptance.__file__).resolve().parent.parent)
                manifest = json.loads((directory / "manifest.json").read_text())
                baseline, current = manifest["cases"]
                self.assertEqual(baseline["status"], "skipped")
                self.assertNotIn("report", baseline)
                self.assertEqual(current["status"], "measured")
                self.assertIn("Comparison unavailable", output.getvalue())
                self.assertIn("current.png", output.getvalue())
                self.assertTrue(directory.with_suffix(".zip").is_file())


class VisualWorkerTests(unittest.TestCase):
    def test_shared_driver_uses_real_loop_hooks_one_wave_and_final_capture(self):
        # Fake application objects solely exercise hooks, without GPU imports.
        settings = SimpleNamespace(with_updates=lambda **kwargs: settings)
        camera = SimpleNamespace(azimuth=1, elevation=2, distance=3, target=(0, 0, 0))
        simulation = SimpleNamespace(simulated_time=0, discarded_time=0, paused=False,
                                     last_error=None, applied_settings=settings, launch_wave=Mock())
        controls = SimpleNamespace(update=Mock(return_value=True), quit_requested=False,
                                   settings=settings, camera=camera, last_error=None)
        window = SimpleNamespace(show=Mock(), save_image=Mock(), destroy=Mock())
        panel = SimpleNamespace(fps=20)
        app = SimpleNamespace(simulation=simulation, controls=controls, panel=panel, warm_up=Mock())
        original_warmup = app.warm_up

        def run_loop():
            app.warm_up()
            for _ in range(20):
                controls.update(window)
                simulation.simulated_time += 0.1
                simulation.discarded_time += 0.05
                window.show()
                if controls.quit_requested:
                    return
            self.fail("driver did not stop the application")

        app.run = run_loop
        initialize, application = Mock(), Mock(return_value=app)
        modules = {"taichi": SimpleNamespace(reset=Mock()),
                   "fluids.app": SimpleNamespace(Application=application, create_window=lambda: window),
                   "fluids.bootstrap": SimpleNamespace(initialize_gpu=initialize),
                   "fluids.config": SimpleNamespace(DEFAULT_SETTINGS=settings)}
        with patch.dict(acceptance.sys.modules, modules), patch.object(
                acceptance, "monotonic", side_effect=range(30)), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(acceptance.visual_worker(["--warmup", "2", "--seconds", "3", "--resets", "0",
                                                      "--capture", "/mock/current.png"]), 0)
        initialize.assert_called_once()
        application.assert_called_once_with(window, settings=settings)
        original_warmup.assert_called_once()
        simulation.launch_wave.assert_called_once_with()
        window.save_image.assert_called_once_with("/mock/current.png")
        window.destroy.assert_called_once()
        modules["taichi"].reset.assert_called_once()
        report = acceptance.read_report(output.getvalue())
        acceptance.validate_report(acceptance.build_cases(warmup_seconds=2, sample_seconds=3)[0], report)
        self.assertEqual(report["waves"], 1)
        self.assertEqual(report["panel_fps"], 20)
        self.assertAlmostEqual(report["simulation_end"]["simulated_seconds"] -
                               report["simulation_start"]["simulated_seconds"], 0.2)

    def test_visual_execution_uses_selected_checkout_and_requires_capture(self):
        case = acceptance.build_cases()[0]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def harness(command, stdout, stderr, **kwargs):
                report = render_fixture(0)
                report["warmup_seconds"] = 2
                report["timing"]["sample_seconds"] = 25
                stdout.write(json.dumps(report))
                (directory / "baseline.png").write_bytes(b"mock capture; no rendered pixels")
                return SimpleNamespace(returncode=0)

            with patch.object(acceptance.subprocess, "run", side_effect=harness) as process:
                result = acceptance.execute_case(case, HARDWARE, directory, checkout=Path("/mock/baseline"))
                self.assertEqual(result["status"], "measured")
                command = process.call_args.args[0]
                self.assertEqual(command[:2], [acceptance.sys.executable, "-c"])
                self.assertIn(str(Path(acceptance.__file__).resolve()), command[2])
                self.assertIn("--visual-worker", command)
                self.assertEqual(process.call_args.kwargs["cwd"], "/mock/baseline")
                self.assertEqual(process.call_args.kwargs["timeout"], 75)
                self.assertEqual(result["capture"], str(directory / "baseline.png"))

    def test_visual_incomplete_panel_sample_and_missing_capture_stay_unmeasured(self):
        case = acceptance.build_cases()[0]
        report = render_fixture(0)
        report["warmup_seconds"] = 2
        report["timing"]["sample_seconds"] = 25
        for key, value in (("panel_fps", 0), ("panel_fps", float("nan")), ("sample_complete", False)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                acceptance.validate_report(case, dict(report, **{key: value}))

        def harness(command, stdout, stderr, **kwargs):
            stdout.write(json.dumps(report))
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance.subprocess, "run", side_effect=harness):
            result = acceptance.execute_case(case, HARDWARE, Path(tmp), checkout=Path("/mock/baseline"))
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("report", result)
            self.assertNotIn("capture", result)
            self.assertIn("PNG", result["reason"])

    def test_deadline_caps_subprocess_timeout(self):
        with patch.object(acceptance, "monotonic", return_value=100):
            self.assertEqual(acceptance.process_timeout(103, 75), 3)
            self.assertEqual(acceptance.process_timeout(99, 75), 0.1)
            self.assertEqual(acceptance.process_timeout(None, 75), 75)


class SummaryAndArchiveTests(unittest.TestCase):
    def setUp(self):
        baseline = patch.object(acceptance, "baseline_worktree", side_effect=lambda *a, **k:
                                contextlib.nullcontext((Path("/mock/baseline"), None)))
        self.baseline = baseline.start()
        self.addCleanup(baseline.stop)
        probe = patch.object(acceptance, "probe_issue_39", return_value=None)
        self.probe = probe.start()
        self.addCleanup(probe.stop)

    def test_no_flags_runs_baseline_then_current_and_prints_comparison(self):
        def measured(case, hardware, directory, **kwargs):
            report = render_fixture(0)
            report["warmup_seconds"] = 2
            report["timing"]["sample_seconds"] = 25
            report["panel_fps"] = 12 if case.name == "baseline" else 18
            if case.name == "current":
                report["simulation_end"]["simulated_seconds"] = 3.5
            return {"name": case.name, "issue": 46, "graphical": True, "status": "measured", "report": report,
                    "capture": str(directory / f"{case.name}.png")}

        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance.Path, "cwd", return_value=Path(tmp)), \
                patch.object(acceptance, "preflight", return_value=HARDWARE), patch.object(
                    acceptance, "execute_case", side_effect=measured) as execute, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(acceptance.main([]), 0)
            self.probe.assert_not_called()
            self.assertEqual([call.args[0].name for call in execute.call_args_list], ["baseline", "current"])
            self.assertEqual(execute.call_args_list[0].kwargs["checkout"], Path("/mock/baseline"))
            self.assertEqual(execute.call_args_list[1].kwargs["checkout"], Path(acceptance.__file__).resolve().parent.parent)
            self.assertIsNotNone(execute.call_args_list[0].kwargs["deadline"])
            self.assertEqual(execute.call_args_list[0].kwargs["deadline"], execute.call_args_list[1].kwargs["deadline"])
            summary = output.getvalue()
            self.assertIn("Metric | Baseline (1c3132b) | Current", summary)
            self.assertIn("panel FPS | 12 | 18", summary)
            self.assertIn("sim rate | 0.01 | 0.02", summary)
            self.assertIn("discard rate | 0.005 | 0.005", summary)
            self.assertIn("Current minus baseline: panel FPS=6; sim rate=0.01", summary)
            self.assertIn("baseline.png", summary)
            self.assertIn("current.png", summary)
            self.assertNotIn("WARNING", summary)
            manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
            self.assertEqual((manifest["warmup_seconds"], manifest["sample_seconds"]), (2, 25))
            self.assertEqual(manifest["plan"]["cases"], 2)
            self.assertTrue(list(Path(tmp).glob("*.zip")))

    def test_no_flags_preflight_failure_never_creates_baseline_and_archives_both_skips(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance.Path, "cwd", return_value=Path(tmp)), \
                patch.object(acceptance, "preflight", side_effect=acceptance.PreflightError("headless")), \
                patch.object(acceptance, "execute_case") as execute, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(acceptance.main([]), 1)
            self.baseline.assert_not_called()
            self.probe.assert_not_called()
            execute.assert_not_called()
            manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
            self.assertEqual([row["name"] for row in manifest["cases"]], ["baseline", "current"])
            self.assertTrue(all(row["status"] == "skipped" and "report" not in row for row in manifest["cases"]))
            self.assertIn("unexecuted/unmeasured", output.getvalue())
            self.assertTrue(list(Path(tmp).glob("*.zip")))

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
            self.assertEqual(len(manifest["cases"]), 3)
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

    def test_plan_is_printed_before_preflight_and_derived_from_actual_case_list(self):
        cases = acceptance.build_cases(37, warmup_seconds=1, sample_seconds=2)[:2]
        output = io.StringIO()

        def refuse(directory, **kwargs):
            self.assertIn("2 cases (2 graphical, 0 stage diagnostics), 0 resets", output.getvalue())
            self.assertIn("Warm-up/sampling alone: 0.1 min", output.getvalue())
            self.assertIn("Estimated duration: 0.8–2.1 min", output.getvalue())
            self.assertIn("derived from the selected case list", output.getvalue())
            raise acceptance.PreflightError("headless")

        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "build_cases", return_value=cases), \
                patch.object(acceptance, "preflight", side_effect=refuse), contextlib.redirect_stdout(output):
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory)]), 1)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["plan"], acceptance.plan_cases(cases))
            self.assertIn("Planned default profile: 2 cases", (directory / "summary.txt").read_text())

    def test_main_default_and_full_pass_timing_options_and_archive_evidence_warning(self):
        def measured(case, hardware, directory, **kwargs):
            report = render_fixture(acceptance.reset_count(case)) if case.capture else native_fixture()
            if case.graphical:
                warmup, sample = acceptance.graphical_window(case)
                report["warmup_seconds"] = warmup
                (report["timing"] if case.capture else report)["sample_seconds"] = sample
            else:
                report = {"kind": "synchronized_evolving_stages"}
            acceptance.validate_report(case, report)
            return {"name": case.name, "issue": case.issue, "graphical": case.graphical,
                    "status": "measured" if case.graphical else "diagnostic", "report": report}

        for full in (False, True):
            with self.subTest(full=full), tempfile.TemporaryDirectory() as tmp, patch.object(
                    acceptance, "preflight", return_value=HARDWARE), patch.object(
                    acceptance, "execute_case", side_effect=measured) as run, contextlib.redirect_stdout(
                    io.StringIO()) as stdout:
                directory = Path(tmp) / "run"
                args = ["--results", str(directory), "--warmup-seconds", "1", "--sample-seconds", "3"]
                if full:
                    args.append("--full")
                self.assertEqual(acceptance.main(args), 0)
                cases = acceptance.build_cases(full=full, warmup_seconds=1, sample_seconds=3)
                self.assertEqual([call.args[0] for call in run.call_args_list], cases)
                self.assertIn("graphical (1s warm-up + 3s sample)", stdout.getvalue())
                self.assertIn("Shorter windows weaken evidence", stdout.getvalue())
                self.assertIn("does not meet the issues' acceptance requirement", stdout.getvalue())
                summary = (directory / "summary.txt").read_text()
                self.assertIn("quick re-check", summary)
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["profile"], "full" if full else "default")
                self.assertEqual((manifest["warmup_seconds"], manifest["sample_seconds"]), (1, 3))
                self.assertEqual(manifest["plan"], acceptance.plan_cases(cases))

    def test_invalid_timing_options_stop_before_creating_results_or_probing(self):
        for flag in ("--sample-seconds", "--warmup-seconds"):
            for value in ("0", "-1", "nan", "inf", "-inf", "invalid"):
                with self.subTest(flag=flag, value=value), tempfile.TemporaryDirectory() as tmp, \
                        contextlib.redirect_stderr(io.StringIO()):
                    directory = Path(tmp) / "run"
                    with self.assertRaises(SystemExit) as exit_status:
                        acceptance.main(["--results", str(directory), f"{flag}={value}"])
                    self.assertEqual(exit_status.exception.code, 2)
                    self.assertFalse(directory.exists())
                    self.probe.assert_not_called()

    def test_optional_checks_are_non_performance_and_failure_blocks_cases(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", return_value=HARDWARE), patch.object(
                acceptance, "run_checks", return_value=[{"name": "CPU suite", "status": "failed"}]), patch.object(
                acceptance, "execute_case") as run, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory), "--with-checks"]), 1)
            run.assert_not_called()

    def test_only_39_runs_all_available_cases_and_separates_stage_diagnostics(self):
        def measured(case, hardware, directory, **kwargs):
            return {"name": case.name, "issue": case.issue, "graphical": case.graphical,
                    "status": "measured" if case.graphical else "diagnostic",
                    "report": native_fixture() if case.graphical else
                    {"kind": "synchronized_evolving_stages", "simulated_seconds": 0.1}}

        with tempfile.TemporaryDirectory() as tmp, patch.object(acceptance, "preflight", return_value=HARDWARE), patch.object(
                acceptance, "execute_case", side_effect=measured) as run, contextlib.redirect_stdout(io.StringIO()) as stdout:
            directory = Path(tmp) / "run"
            self.assertEqual(acceptance.main(["--results", str(directory), "--only", "39", "--full"]), 0)
            self.assertEqual([call.args[0] for call in run.call_args_list], acceptance.build_cases(39, full=True))
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

        def measured(case, hardware, directory, **kwargs):
            return {"name": case.name, "issue": case.issue, "graphical": case.graphical,
                    "status": "measured" if case.graphical else "diagnostic",
                    "report": render_fixture() if case.capture else native_fixture() if case.graphical else {}}

        for only, expected, size in ((["--full"], 0, 33), (["--only", "39"], 1, 0)):
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
            self.assertEqual(len(rows), 2)
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
