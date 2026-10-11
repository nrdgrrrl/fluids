"""Check the diagnostic on CPU; this does not establish Vulkan compatibility."""

import contextlib
import io
import unittest
from unittest.mock import patch

import taichi as ti

from fluids import smoke


class SmokeTests(unittest.TestCase):
    def test_smoke_invariants_on_cpu(self):
        ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=False)
        self.addCleanup(ti.reset)
        with contextlib.redirect_stdout(io.StringIO()):
            report = smoke.run_checks(steps=2)
        self.assertEqual(report["backend"], str(ti.cpu))
        self.assertEqual(report["quality"], "medium")
        self.assertEqual(report["particles"], 18_360)
        self.assertEqual(report["water_city_resets"], 10)

    def test_cli_reports_unsupported_gpu_without_running_checks(self):
        stderr = io.StringIO()
        with (
            patch.object(smoke, "initialize_gpu", side_effect=RuntimeError("unsupported Vulkan")),
            patch.object(smoke, "run_checks") as checks,
            patch.object(ti, "reset"),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(smoke.main([]), 1)
        checks.assert_not_called()
        self.assertIn("FAIL: unsupported Vulkan", stderr.getvalue())

    def test_cli_rejects_zero_steps_before_initialization(self):
        with (
            patch.object(smoke, "initialize_gpu") as initialize,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exit_status,
        ):
            smoke.main(["--steps", "0"])
        self.assertEqual(exit_status.exception.code, 2)
        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
