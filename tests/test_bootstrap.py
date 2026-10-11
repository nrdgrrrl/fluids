"""Startup and lifecycle checks using mocks, with no GPU or display access."""

import contextlib
import io
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from fluids import bootstrap
from fluids.__main__ import main


class BootstrapTests(unittest.TestCase):
    def test_package_import_does_not_load_taichi(self):
        result = subprocess.run(
            [sys.executable, "-c", "import fluids, sys; assert 'taichi' not in sys.modules"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_vulkan_initialization_disables_fallback(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(bootstrap.ti, "init") as initialize,
            patch.object(bootstrap.ti, "cfg", SimpleNamespace(arch=bootstrap.ti.vulkan)),
        ):
            bootstrap.initialize_gpu()
        initialize.assert_called_once_with(
            arch=bootstrap.ti.vulkan, enable_fallback=False, offline_cache=True
        )

    def test_backend_overrides_are_rejected_before_initialization(self):
        for arch in ("x64", "cuda", "", "gpu"):
            with (
                self.subTest(arch=arch),
                patch.dict(os.environ, {"TI_ARCH": arch}),
                patch.object(bootstrap.ti, "init") as initialize,
            ):
                with self.assertRaisesRegex(RuntimeError, "Vulkan initialization failed") as raised:
                    bootstrap.initialize_gpu()
                self.assertIn("Unset TI_ARCH", str(raised.exception.__cause__))
                initialize.assert_not_called()

    def test_unexpected_backend_is_reset_and_rejected(self):
        with (
            patch.dict(os.environ, {"TI_ARCH": "vulkan"}),
            patch.object(bootstrap.ti, "init"),
            patch.object(bootstrap.ti, "cfg", SimpleNamespace(arch=bootstrap.ti.cpu)),
            patch.object(bootstrap.ti, "reset") as reset,
        ):
            with self.assertRaisesRegex(RuntimeError, "Vulkan initialization failed"):
                bootstrap.initialize_gpu()
            reset.assert_called_once_with()

    def test_gpu_failure_is_actionable_and_does_not_create_window(self):
        error = RuntimeError("Arch=vulkan is not supported")
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {"TI_ARCH": "vulkan"}),
            patch.object(bootstrap.ti, "init", side_effect=error),
            patch.object(bootstrap, "create_window") as create,
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(main(), 1)
        create.assert_not_called()
        self.assertIn(bootstrap.VULKAN_ERROR, stderr.getvalue())
        self.assertIn(str(error), stderr.getvalue())

    def test_window_failure_is_actionable(self):
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {"DISPLAY": ":0"}),
            patch.object(bootstrap, "initialize_gpu"),
            patch.object(bootstrap.ti.ui, "Window", side_effect=RuntimeError("GGUI unavailable")),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(main(), 1)
        self.assertIn("Native window initialization failed", stderr.getvalue())
        self.assertIn("GGUI unavailable", stderr.getvalue())

    def test_headless_linux_is_rejected_before_native_window_creation(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(bootstrap.sys, "platform", "linux"),
            patch.object(bootstrap.ti.ui, "Window") as create,
        ):
            with self.assertRaisesRegex(RuntimeError, "desktop graphics session with DISPLAY"):
                bootstrap.create_window()
            create.assert_not_called()

    def test_closing_window_ends_loop_and_destroys_window(self):
        window = Mock()
        type(window).running = PropertyMock(side_effect=[True, False])
        with (
            patch.dict(os.environ, {"DISPLAY": ":0"}),
            patch.object(bootstrap, "initialize_gpu"),
            patch.object(bootstrap.ti.ui, "Window", return_value=window),
        ):
            self.assertEqual(main(), 0)
        window.show.assert_called_once_with()
        window.destroy.assert_called_once_with()

    def test_interrupt_still_destroys_window(self):
        window = Mock()
        window.show.side_effect = KeyboardInterrupt
        with (
            patch.object(bootstrap, "initialize_gpu"),
            patch.object(bootstrap, "create_window", return_value=window),
        ):
            self.assertEqual(main(), 130)
        window.destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
