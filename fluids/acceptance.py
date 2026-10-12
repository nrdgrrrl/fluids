"""One-command desktop acceptance; imports only the Python standard library."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


class PreflightError(RuntimeError):
    """The host cannot supply trustworthy graphical evidence."""


@dataclass(frozen=True)
class Hardware:
    renderer: str
    gpu: str
    driver: str
    refresh_hz: float
    displays: dict


@dataclass(frozen=True)
class Case:
    name: str
    issue: int
    module: str
    flags: tuple
    graphical: bool = True
    capture: bool = False


def require_host(version, environ, devices):
    if tuple(version[:2]) != (3, 11):
        raise PreflightError("Python >=3.11,<3.12 is required. Activate the documented Python 3.11 venv.")
    if not environ.get("DISPLAY", "").strip():
        raise PreflightError("DISPLAY is unset. Run from a terminal in a real X11/XWayland desktop session.")
    if not devices:
        raise PreflightError("No GPU device under /dev/dri. Run on the GPU desktop with its vendor driver.")
    if environ.get("TI_ARCH", "vulkan") != "vulkan":
        raise PreflightError("Unset TI_ARCH or set it to vulkan; acceptance requires the physical Vulkan GPU.")


def refuse_software(text):
    match = re.search(r"llvmpipe|lavapipe|softpipe|swrast|PHYSICAL_DEVICE_TYPE_CPU", text, re.I)
    if match:
        raise PreflightError(f"Software renderer detected ({match.group()}). Select the vendor's physical Vulkan GPU/driver; "
                             "software rasterization cannot supply performance evidence.")


def detect_hardware(glx, vulkan, xrandr):
    """Parse system-tool evidence conservatively, never supplying guessed labels."""
    refuse_software(glx + "\n" + vulkan)
    devices = re.findall(r"^\s*deviceName\s*=\s*(.+)$", vulkan, re.M)
    if len(devices) != 1:
        raise PreflightError("vulkaninfo --summary must identify exactly one physical GPU. "
                             "Select its ICD with VK_DRIVER_FILES (or VK_ICD_FILENAMES) and retry.")
    if not re.search(r"deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(?:INTEGRATED|DISCRETE)_GPU", vulkan):
        raise PreflightError("vulkaninfo did not identify a physical GPU type; check the vendor Vulkan driver.")
    drivers = re.findall(r"^\s*(?:driverName|driverInfo)\s*=\s*(.+)$", vulkan, re.M)
    if not drivers:
        raise PreflightError("vulkaninfo did not identify the driver. Check the vendor Vulkan driver/tools.")
    renderer = re.search(r"^\s*OpenGL renderer string:\s*(.+)$", glx, re.M)
    # Vulkan's deviceName is also its renderer label if GLX is unavailable.
    renderer = renderer.group(1).strip() if renderer else devices[0].strip()
    displays = {}
    current = None
    for line in xrandr.splitlines():
        if line and not line[0].isspace():
            match = re.match(r"(\S+) connected(?:\s|$)", line)
            current = match.group(1) if match else None
        elif current:
            rate = re.search(r"(\d+(?:\.\d+)?)\*", line)
            if rate:
                displays[current] = float(rate.group(1))
    rates = set(displays.values())
    if len(rates) != 1 or not all(isfinite(rate) and rate > 0 for rate in rates):
        raise PreflightError("Cannot determine a single active refresh rate from xrandr --current. "
                             "Use one active display or equal refresh rates on all displays, then retry.")
    return Hardware(renderer, devices[0].strip(), " / ".join(drivers), rates.pop(), displays)


def preflight(directory):
    devices = [path for pattern in ("card*", "renderD*") for path in Path("/dev/dri").glob(pattern)
               if path.is_char_device()]
    require_host(sys.version_info, os.environ, devices)
    evidence = {}
    for name, command, required in (
        ("glxinfo", ["glxinfo", "-B"], False),
        ("vulkaninfo", ["vulkaninfo", "--summary"], True),
        ("xrandr", ["xrandr", "--current"], True),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            (directory / f"{name}.txt").write_text(str(exc) + "\n")
            if required:
                raise PreflightError(f"Cannot read {name}. Ensure the system graphics tools "
                                     f"({name}) are available in this desktop session: {exc}") from exc
            evidence[name] = ""
            continue
        (directory / f"{name}.txt").write_text(result.stdout + "\n" + result.stderr)
        refuse_software(result.stdout)
        if required and result.returncode:
            raise PreflightError(f"{name} failed. Check the desktop/vendor driver; see {name}.txt in results.")
        evidence[name] = result.stdout if result.returncode == 0 else ""
    return detect_hardware(evidence["glxinfo"], evidence["vulkaninfo"], evidence["xrandr"])


def probe_issue_39(directory):
    """Return a truthful skip reason, or None when the budget flag is supported."""
    command = [sys.executable, "-m", "fluids.benchmark", "--help"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        (directory / "benchmark-help.stderr.txt").write_text(str(exc) + "\n")
        return f"Could not probe fluids.benchmark --help: {exc}; see benchmark-help.stderr.txt."
    (directory / "benchmark-help.stdout.txt").write_text(result.stdout)
    (directory / "benchmark-help.stderr.txt").write_text(result.stderr)
    if result.returncode:
        return (f"fluids.benchmark --help exited {result.returncode}; "
                "see benchmark-help.stderr.txt and check the documented venv/dependencies.")
    if not re.search(r"(?<![\w-])--physics-budget-ms(?![\w-])", result.stdout):
        return "fluids.benchmark --help does not advertise --physics-budget-ms in this checkout."
    return None


def build_cases(only=None, *, issue_39_available=True, full=False, warmup_seconds=8, sample_seconds=30):
    """Minimal comparisons by default; --full retains the documented matrices."""
    cases = []
    native = ("--mode", "native", "--backend", "vulkan", "--warmup-seconds", f"{warmup_seconds:g}",
              "--sample-seconds", f"{sample_seconds:g}")
    stages = ("--backend", "vulkan", "--steps", "40", "--warmup-steps", "8")
    for layout in ("legacy", "fixed", "quality"):
        for scenario in ("idle", "wave", "paused"):
            cases.append(Case(f"36-{layout}-{scenario}", 36, "fluids.benchmark",
                              native + ("--layout", layout, "--scenario", scenario)))
        for wave in (False, True):
            cases.append(Case(f"36-{layout}-stages-{'wave' if wave else 'idle'}", 36, "fluids.benchmark",
                              stages + ("--layout", layout) + (("--wave", "--presteps", "120") if wave else ()),
                              graphical=False))
    for scenario in ("paused", "active", "wave"):
        for staging in ("stock", "device"):
            cases.append(Case(f"37-{staging}-{scenario}", 37, "fluids.render_benchmark",
                              ("--staging", staging, "--scenario", scenario, "--quality", "medium",
                               "--warmup", f"{warmup_seconds:g}", "--seconds", f"{sample_seconds:g}",
                               "--resets", "10" if full else "0"), capture=True))
    for reads in ("duplicate", "cached"):
        for scenario, quality in (("idle", "medium"), ("wave", "medium"), ("paused", "medium"),
                                  ("wave", "low")):
            cases.append(Case(f"38-{reads}-{quality}-{scenario}", 38, "fluids.benchmark",
                              native + ("--timestep-reads", reads, "--scenario", scenario, "--quality", quality)))
        for wave in (False, True):
            cases.append(Case(f"38-{reads}-stages-{'wave' if wave else 'idle'}", 38, "fluids.benchmark",
                              stages + ("--timestep-reads", reads) + (("--wave", "--presteps", "120") if wave else ()),
                              graphical=False))
    if issue_39_available:
        for budget in ("12", "0"):
            budget_flags = ("--physics-budget-ms", budget)
            for scenario, quality in (("idle", "medium"), ("wave", "medium"), ("paused", "medium"),
                                      ("wave", "low")):
                cases.append(Case(f"39-budget-{budget}-{quality}-{scenario}", 39, "fluids.benchmark",
                                  native + budget_flags + ("--scenario", scenario, "--quality", quality)))
            for wave in (False, True):
                cases.append(Case(f"39-budget-{budget}-stages-{'wave' if wave else 'idle'}", 39,
                                  "fluids.benchmark", stages + budget_flags +
                                  (("--wave", "--presteps", "120") if wave else ()), graphical=False))
    default_names = {
        "36-legacy-idle", "36-fixed-idle", "36-fixed-wave",
        "38-duplicate-medium-idle", "38-cached-medium-idle",
        "37-stock-paused", "37-device-paused", "37-device-active",
        "39-budget-12-medium-wave", "39-budget-0-medium-wave",
    }
    return [case for case in cases if (only is None or case.issue == only)
            and (full or case.name in default_names)]


def graphical_window(case):
    warmup_flag, sample_flag = (("--warmup", "--seconds") if case.capture
                                else ("--warmup-seconds", "--sample-seconds"))
    return (float(case.flags[case.flags.index(warmup_flag) + 1]),
            float(case.flags[case.flags.index(sample_flag) + 1]))


def reset_count(case):
    return int(case.flags[case.flags.index("--resets") + 1]) if case.capture else 0


def plan_cases(cases):
    """Unmeasured planning allowances, summed from the actual selected cases.

    Allow 20–60s per graphical process for startup/compilation, 30–90s per
    synchronized diagnostic, and 2–15s per reset. Hardware can exceed these.
    Optional correctness checks and preflight are outside this estimate.
    """
    graphical = sum(case.graphical for case in cases)
    diagnostics = len(cases) - graphical
    resets = sum(reset_count(case) for case in cases)
    window_seconds = sum(sum(graphical_window(case)) for case in cases if case.graphical)
    return {"cases": len(cases), "graphical": graphical, "diagnostics": diagnostics, "resets": resets,
            "window_seconds": window_seconds,
            "estimated_min_seconds": window_seconds + 20 * graphical + 30 * diagnostics + 2 * resets,
            "estimated_max_seconds": window_seconds + 60 * graphical + 90 * diagnostics + 15 * resets}


def format_plan(plan, full):
    return (f"Planned {'full' if full else 'default'} profile: {plan['cases']} cases "
            f"({plan['graphical']} graphical, {plan['diagnostics']} stage diagnostics), "
            f"{plan['resets']} resets. Warm-up/sampling alone: {plan['window_seconds'] / 60:.1f} min. "
            f"Estimated duration: {plan['estimated_min_seconds'] / 60:.1f}–"
            f"{plan['estimated_max_seconds'] / 60:.1f} min, derived from the selected case list "
            "with unmeasured startup/compilation, diagnostic and reset allowances; "
            "GPU-dependent, excludes preflight and --with-checks.")


def case_command(case, hardware, directory):
    gpu_flag = "--gpu" if case.capture else "--device-label"
    command = [sys.executable, "-m", case.module, *case.flags, gpu_flag, hardware.gpu,
               "--driver", hardware.driver, "--refresh-hz", str(hardware.refresh_hz)]
    if case.capture:
        command.extend(("--capture", str(directory / f"{case.name}.png"),
                        "--output", str(directory / f"{case.name}.json")))
    return command


def read_report(stdout):
    """Harness startup banners precede the single JSON report on stdout."""
    for match in re.finditer(r"^\s*\{", stdout, re.M):
        try:
            report, _ = json.JSONDecoder().raw_decode(stdout[match.start():].lstrip())
        except ValueError:
            continue
        if isinstance(report, dict):
            return report
    raise ValueError("Harness did not produce a JSON report; inspect its stdout/stderr logs.")


def validate_report(case, report):
    """Require the requested window and reset count, including quick re-checks."""
    if not case.graphical:
        if report.get("kind") != "synchronized_evolving_stages":
            raise ValueError("Expected a synchronized stage diagnostic")
        return
    if case.capture:
        timing = report["timing"]
        warmup = report["warmup_seconds"]
        frame_ms = timing["mean_frame_ms"]
        for boundary in ("simulation_start", "simulation_end"):
            for key in ("simulated_seconds", "discarded_seconds"):
                if not isfinite(report[boundary][key]):
                    raise ValueError("Non-finite simulation counters")
        if len(report["water_city_resets"]) != reset_count(case):
            raise ValueError("Requested reset sweep did not complete")
    else:
        timing = report
        warmup = report["warmup_seconds"]
        frame_ms = report["frame"]["mean_ms"]
        if report.get("kind") != "complete_native_frames" or report.get("sample_complete") is not True:
            raise ValueError("Native sample did not complete; graphical results are unmeasured")
        for key in ("simulated_seconds", "discarded_seconds"):
            if not isfinite(report[key]):
                raise ValueError("Non-finite simulation counters")
    required_warmup, required_sample = graphical_window(case)
    if (not all(isfinite(value) for value in (warmup, timing["sample_seconds"], timing["fps"], frame_ms))
            or warmup < required_warmup or timing["sample_seconds"] < required_sample
            or timing["fps"] <= 0 or frame_ms <= 0):
        raise ValueError(f"Invalid or insufficient warmed graphical sample (requires {required_warmup:g}s "
                         f"warm-up and {required_sample:g}s sample)")


def execute_case(case, hardware, directory):
    command = case_command(case, hardware, directory)
    result = {"name": case.name, "issue": case.issue, "graphical": case.graphical, "command": command,
              "status": "failed", "reason": "unexecuted/unmeasured"}
    try:
        with ((directory / f"{case.name}.stdout.txt").open("w") as stdout,
              (directory / f"{case.name}.stderr.txt").open("w") as stderr):
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=1800)
        if completed.returncode:
            raise ValueError(f"Harness exited {completed.returncode}; inspect {case.name}.stderr.txt")
        report = read_report((directory / f"{case.name}.stdout.txt").read_text())
        validate_report(case, report)
        if case.capture and not (directory / f"{case.name}.png").is_file():
            raise ValueError("Harness did not save the required PNG capture")
        report["acceptance_hardware"] = asdict(hardware)
        (directory / f"{case.name}.json").write_text(json.dumps(report, indent=2) + "\n")
        result.update(status="measured" if case.graphical else "diagnostic", report=report)
        result.pop("reason")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        result["reason"] = str(exc)
    return result


def _number(value):
    return "unmeasured" if value is None else f"{value:.6g}"


def _flatten(value, prefix=""):
    """Keep every reported counter/timing available in copyable plain text."""
    if isinstance(value, dict):
        return "; ".join(_flatten(item, f"{prefix}.{key}" if prefix else key) for key, item in value.items())
    if isinstance(value, list):
        # Reset sweeps share metadata. Transpose rows into short counter lists
        # instead of printing the same field names/labels ten times per case.
        if value and all(isinstance(item, dict) and item.keys() == value[0].keys() for item in value):
            return _flatten({key: [item[key] for item in value] for key in value[0]}, prefix)
        if value and all(item == value[0] for item in value):
            return _flatten(value[0], f"{prefix}[all {len(value)}]")
        return f"{prefix}=[" + ",".join(_number(item) if isinstance(item, float) else str(item) for item in value) + "]"
    return f"{prefix}={_number(value) if isinstance(value, float) else value}"


def format_summary(results, hardware=None, checks=(), note=None, issue_39=None, plan_text=None):
    lines = ["Fluids acceptance (plain text)"]
    if plan_text:
        lines.append(plan_text)
    if hardware:
        lines.append(f"Renderer: {hardware.renderer}; GPU: {hardware.gpu}; driver: {hardware.driver}; "
                     f"refresh: {hardware.refresh_hz:g} Hz ({_flatten(hardware.displays)})")
    host_report = next((result["report"] for result in results if "report" in result), {})
    host_labels = {key: host_report[key] for key in ("os", "python", "taichi") if key in host_report}
    if host_labels:
        lines.append("Host: " + _flatten(host_labels))
    if note:
        lines.append(note)
    if issue_39 and issue_39["status"] == "skipped":
        lines.append(f"Issue #39: skipped — {issue_39['reason']}")
    lines.append("Correctness prerequisites — NON-PERFORMANCE (no frame/FPS evidence):")
    lines.extend(f"  {check['name']}: {check['status']}" for check in checks)
    if not checks:
        lines.append("  skipped (enable with --with-checks)")
    for graphical, heading in ((True, "GRAPHICAL native frames"),
                               (False, "SYNCHRONIZED Vulkan stage diagnostics — not frame/FPS measurements")):
        lines.append(heading + ":")
        rows = [result for result in results if result["graphical"] == graphical]
        if not rows:
            lines.append("  none selected; unexecuted/unmeasured")
        for result in rows:
            if result["status"] not in ("measured", "diagnostic"):
                state = "unexecuted/unmeasured" if result["status"] == "skipped" else "unmeasured"
                lines.append(f"  {result['name']}: {result['status']} — {state}: {result['reason']}")
                continue
            report = result["report"]
            if graphical:
                if "timing" in report:
                    fps, frame = report["timing"]["fps"], report["timing"]["mean_frame_ms"]
                    simulated, discarded = (report["simulation_end"][key] - report["simulation_start"][key]
                                             for key in ("simulated_seconds", "discarded_seconds"))
                else:
                    fps, frame = report["fps"], report["frame"]["mean_ms"]
                    simulated, discarded = report["simulated_seconds"], report["discarded_seconds"]
                lines.append(f"  {result['name']}: measured graphically; FPS={_number(fps)}; "
                             f"frame_ms={_number(frame)}; simulated_s={_number(simulated)}; "
                             f"discarded_s={_number(discarded)}")
            else:
                lines.append(f"  {result['name']}: diagnostic; FPS/frame=unmeasured; "
                             f"simulated_s={_number(report.get('simulated_seconds'))}; discarded_s=unreported")
            # Exclude verbose scene settings/host labels; preserve all measured
            # distributions, counters, staging metadata and reset/RSS samples.
            details = {key: value for key, value in report.items() if key not in
                       ("physics_settings", "acceptance_hardware", "os", "python", "taichi", "gpu",
                        "device_label", "driver", "refresh_hz", "first_frame_capture")}
            settings = report.get("physics_settings", {})
            for key in ("quality", "water_depth"):
                if key in settings:
                    details[key] = settings[key]
            lines.append("    " + _flatten(details))
    lines.append("GPU memory/render-engine activity and human visual/interaction acceptance remain owner checks; "
                 "RSS and logical buffer bytes are not GPU memory.")
    return "\n".join(lines) + "\n"


def assemble_archive(directory, manifest, summary):
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "summary.txt").write_text(summary)
    return Path(shutil.make_archive(str(directory), "zip", root_dir=directory.parent, base_dir=directory.name))


def run_checks(directory):
    """Optional correctness only; never use their clocks as performance data."""
    checks = []
    environment = dict(os.environ, FLUIDS_TEST_PARTICLE_ARCH="cpu", FLUIDS_TEST_NATIVE="0")
    environment.pop("TI_ARCH", None)
    for name, command in (
        ("focused CPU unittest", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_core.py", "-v"]),
        ("full CPU unittest", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("windowless Vulkan smoke", ["-m", "fluids.smoke"]),
    ):
        print(f"NON-PERFORMANCE prerequisite: {name}", flush=True)
        if "unittest" in command and not Path("tests/test_core.py").is_file():
            checks.append({"name": name, "status": "failed (run --with-checks from the repository root)"})
            break
        with (directory / f"check-{len(checks) + 1}.txt").open("w") as log:
            try:
                completed = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT,
                                           env=environment, timeout=1800)
                status = "passed" if completed.returncode == 0 else f"failed (exit {completed.returncode})"
            except (OSError, subprocess.TimeoutExpired) as exc:
                status = f"failed ({exc})"
        checks.append({"name": name, "status": status})
        if status != "passed":
            break
    return checks


def positive_seconds(value):
    seconds = float(value)
    if not isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=int, choices=(36, 37, 38, 39), help="run only this issue's cases")
    parser.add_argument("--full", action="store_true", help="run exhaustive matrices, stage diagnostics and reset sweeps")
    parser.add_argument("--sample-seconds", type=positive_seconds, default=30,
                        help="graphical sample window (default: 30s); shorter windows weaken evidence")
    parser.add_argument("--warmup-seconds", type=positive_seconds, default=8,
                        help="graphical warm-up (default: 8s); shorter warm-up weakens evidence")
    parser.add_argument("--results", type=Path, help="new results directory (must not already exist)")
    parser.add_argument("--with-checks", action="store_true", help="first run non-performance CPU tests and GPU smoke")
    args = parser.parse_args(argv)
    if args.results:
        directory = args.results.resolve()
        if directory.exists() or Path(str(directory) + ".zip").exists():
            parser.error("--results directory or its ZIP already exists; choose a new path for this pass")
        try:
            directory.mkdir(parents=True)
        except OSError as exc:
            parser.error(f"Cannot create --results directory: {exc}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = Path(tempfile.mkdtemp(prefix=f"fluids-acceptance-{stamp}-", dir=Path.cwd()))
    if args.only in (None, 39):
        reason = probe_issue_39(directory)
        issue_39 = {"status": "skipped", "reason": reason} if reason else {"status": "available"}
    else:
        issue_39 = {"status": "not selected"}
    cases = build_cases(args.only, issue_39_available=issue_39["status"] == "available", full=args.full,
                        warmup_seconds=args.warmup_seconds, sample_seconds=args.sample_seconds)
    plan = plan_cases(cases)
    plan_text = format_plan(plan, args.full)
    if args.sample_seconds < 30 or args.warmup_seconds < 8:
        plan_text += ("\nWARNING: Shorter windows weaken evidence; this is a quick re-check. "
                      "A sample below 30s does not meet the issues' acceptance requirement; "
                      "repeat with the default 8s warm-up and 30s sample for acceptance.")
    print(plan_text, flush=True)
    results = [{"name": case.name, "issue": case.issue, "graphical": case.graphical,
                "status": "skipped", "reason": "not run"} for case in cases]
    hardware, checks, note, active_index = None, [], None, None
    try:
        if not cases:
            raise PreflightError(f"No cases available for issue #39: {issue_39['reason']}")
        hardware = preflight(directory)
        print(f"Renderer: {hardware.renderer}\nGPU: {hardware.gpu}\nDriver: {hardware.driver}\n"
              f"Refresh: {hardware.refresh_hz:g} Hz\nResults: {directory}", flush=True)
        print("Do not use the mouse or keyboard during the pass. Every case starts a fresh process.", flush=True)
        if args.with_checks:
            checks = run_checks(directory)
            if any(check["status"] != "passed" for check in checks):
                raise PreflightError("Non-performance correctness prerequisite failed; see check logs.")
        for index, case in enumerate(cases):
            active_index = index
            timing = ("graphical (%gs warm-up + %gs sample)" % graphical_window(case) if case.graphical
                      else "stage diagnostic; no FPS")
            print(f"[{index + 1}/{len(cases)}] {case.name}: "
                  f"{timing}",
                  flush=True)
            results[index] = execute_case(case, hardware, directory)
            active_index = None
            print(f"  {results[index]['status']}", flush=True)
    except (PreflightError, KeyboardInterrupt) as exc:
        note = f"Acceptance stopped: {exc or 'interrupted by owner'}; remaining cases unexecuted/unmeasured."
        if active_index is not None:
            results[active_index].update(status="failed", reason="interrupted; graphical sample unmeasured")
        for result in results:
            if result["status"] == "skipped":
                result["reason"] = note
    summary = format_summary(results, hardware, checks, note, issue_39, plan_text)
    manifest = {"hardware": asdict(hardware) if hardware else None, "only": args.only,
                "profile": "full" if args.full else "default", "plan": plan,
                "warmup_seconds": args.warmup_seconds, "sample_seconds": args.sample_seconds,
                "checks_non_performance": checks, "cases": results, "note": note,
                "issue_39": issue_39}
    archive = assemble_archive(directory, manifest, summary)
    print(summary, end="")
    print(f"Results archive: {archive}", flush=True)
    return int(bool(note) or not cases or any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    sys.exit(main())
