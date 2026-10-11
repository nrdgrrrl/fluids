"""Startup and lifecycle checks using mocks, with no GPU or display access."""

import contextlib
import io
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from fluids import app, bootstrap
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
            patch.object(app, "create_window") as create,
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
            patch.object(app.ti.ui, "Window", side_effect=RuntimeError("GLFW unavailable")),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(main(), 1)
        self.assertIn("Native window initialization failed", stderr.getvalue())
        self.assertIn("GLFW unavailable", stderr.getvalue())

    def test_headless_linux_is_rejected_before_native_window_creation(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(bootstrap.sys, "platform", "linux"),
            patch.object(bootstrap, "VulkanWindow") as create,
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
            patch.object(app, "create_window", return_value=window),
            patch.object(app, "Application") as application,
        ):
            application.return_value.run.side_effect = window.show
            self.assertEqual(main(), 0)
        application.assert_called_once_with(window)
        window.show.assert_called_once_with()
        window.destroy.assert_called_once_with()

    def test_interrupt_still_destroys_window(self):
        window = Mock()
        window.show.side_effect = KeyboardInterrupt
        with (
            patch.object(bootstrap, "initialize_gpu"),
            patch.object(app, "create_window", return_value=window),
            patch.object(app, "Application") as application,
        ):
            application.return_value.run.side_effect = window.show
            self.assertEqual(main(), 130)
        window.destroy.assert_called_once_with()


class VulkanWindowTests(unittest.TestCase):
    def setUp(self):
        self.glfw = self.enterContext(patch.object(bootstrap, "glfw"))
        self.gpu = self.enterContext(patch.object(bootstrap.wgpu, "gpu"))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch.object(bootstrap.sys, "platform", "linux"))
        self.glfw.get_framebuffer_size.return_value = (1440, 900)
        self.glfw.window_should_close.return_value = False
        self.adapter = self.gpu.request_adapter_sync.return_value
        self.adapter.info = {"backend_type": "Vulkan"}
        self.context = self.gpu.get_canvas_context.return_value
        self.device = self.adapter.request_device_sync.return_value

    def test_creation_enables_resizing_and_requests_vulkan(self):
        def request(**kwargs):
            self.assertEqual(os.environ["WGPU_BACKEND_TYPE"], "Vulkan")
            return self.adapter

        self.gpu.request_adapter_sync.side_effect = request
        window = bootstrap.VulkanWindow()
        self.addCleanup(window.destroy)
        self.glfw.window_hint.assert_any_call(self.glfw.RESIZABLE, self.glfw.TRUE)
        self.glfw.window_hint.assert_any_call(self.glfw.CLIENT_API, self.glfw.NO_API)
        self.gpu.request_adapter_sync.assert_called_once_with(
            canvas=self.context, power_preference="high-performance",
            force_fallback_adapter=False,
        )
        self.assertTrue(self.gpu.get_canvas_context.call_args.args[0]["vsync"])
        self.assertNotIn("WGPU_BACKEND_TYPE", os.environ)

    def test_resized_frames_update_surface_before_rendering(self):
        window = bootstrap.VulkanWindow()
        self.addCleanup(window.destroy)
        sizes = [(640, 480), (1920, 1080), (1440, 900)]
        self.glfw.get_framebuffer_size.side_effect = sizes
        events = []
        self.context.set_physical_size.side_effect = lambda *size: events.append(size)
        self.context.get_current_texture.side_effect = lambda: events.append("draw") or Mock()
        self.context.present.side_effect = lambda: events.append("present")
        for _ in sizes:
            window.show()
        self.assertEqual(events, [event for size in sizes for event in (size, "draw", "present")])
        self.assertEqual(self.device.queue.submit.call_count, 3)

    def test_minimize_skips_rendering_and_restore_resumes(self):
        window = bootstrap.VulkanWindow()
        self.addCleanup(window.destroy)
        self.glfw.get_framebuffer_size.side_effect = [(0, 0), (800, 600)]
        window.show()
        self.context.get_current_texture.assert_not_called()
        self.glfw.wait_events_timeout.assert_called_once_with(0.05)
        window.show()
        self.context.set_physical_size.assert_called_with(800, 600)
        self.context.present.assert_called_once_with()

    def test_non_vulkan_renderer_is_rejected_and_window_cleaned_up(self):
        self.adapter.info = {"backend_type": "OpenGL"}
        with self.assertRaisesRegex(RuntimeError, "requires Vulkan"):
            bootstrap.VulkanWindow()
        self.adapter.request_device_sync.assert_not_called()
        self.context.unconfigure.assert_called_once_with()
        self.glfw.destroy_window.assert_called_once_with(self.glfw.create_window.return_value)
        self.glfw.terminate.assert_called_once_with()

    def test_renderer_override_is_rejected(self):
        os.environ["WGPU_BACKEND_TYPE"] = "OpenGL"
        with self.assertRaisesRegex(RuntimeError, "Unset WGPU_BACKEND_TYPE"):
            bootstrap.VulkanWindow()
        self.gpu.request_adapter_sync.assert_not_called()
        self.assertEqual(os.environ["WGPU_BACKEND_TYPE"], "OpenGL")
        self.glfw.terminate.assert_called_once_with()

    def test_renderer_failure_restores_environment_and_releases_window(self):
        self.gpu.request_adapter_sync.side_effect = RuntimeError("No Vulkan adapter")
        with self.assertRaisesRegex(RuntimeError, "No Vulkan adapter"):
            bootstrap.VulkanWindow()
        self.assertNotIn("WGPU_BACKEND_TYPE", os.environ)
        self.glfw.destroy_window.assert_called_once_with(self.glfw.create_window.return_value)
        self.glfw.terminate.assert_called_once_with()


@unittest.skipUnless(os.environ.get("FLUIDS_TEST_NATIVE") == "1", "requires a Vulkan desktop")
class NativeResizeTests(unittest.TestCase):
    def test_native_resize_renders_at_new_framebuffer_size(self):
        """Exercise the real native window and read back Vulkan-rendered pixels."""
        import time
        import numpy as np

        bootstrap.initialize_gpu()
        self.addCleanup(bootstrap.ti.reset)
        window = bootstrap.create_window()
        self.addCleanup(window.destroy)
        self.assertEqual(
            bootstrap.glfw.get_window_attrib(window._window, bootstrap.glfw.RESIZABLE),
            bootstrap.glfw.TRUE,
        )
        context, device = window._context, window._device
        self.assertEqual(device.adapter.info["backend_type"], "Vulkan")
        context.configure(
            device=device, format=context.get_preferred_format(device.adapter),
            usage=bootstrap.wgpu.TextureUsage.RENDER_ATTACHMENT | bootstrap.wgpu.TextureUsage.COPY_SRC,
        )
        window.set_background_color((0.0, 1.0, 0.0))
        present = context.present
        get_texture = context.get_current_texture
        frames = []

        def read_frame():
            texture = get_texture()
            width, height, _ = texture.size
            data = device.queue.read_texture(
                {"texture": texture}, {"bytes_per_row": width * 4, "rows_per_image": height},
                (width, height, 1),
            )
            pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
            # Green is identical in RGBA/BGRA and in linear/sRGB formats.
            np.testing.assert_array_equal(pixels[0, 0], [0, 255, 0, 255])
            np.testing.assert_array_equal(pixels[-1, -1], [0, 255, 0, 255])
            frames.append((width, height))
            present()

        with patch.object(context, "present", side_effect=read_frame):
            for size in [(1440, 900), (640, 480), (1600, 1000), (800, 600)]:
                bootstrap.glfw.set_window_size(window._window, *size)
                deadline = time.monotonic() + 2
                while bootstrap.glfw.get_window_size(window._window) != size:
                    bootstrap.glfw.poll_events()
                    self.assertLess(time.monotonic(), deadline, "native resize timed out")
                self.assertTrue(window.running)
                framebuffer_size = bootstrap.glfw.get_framebuffer_size(window._window)
                window.show()
                self.assertEqual(frames[-1], framebuffer_size)
        bootstrap.glfw.set_window_should_close(window._window, True)
        self.assertFalse(window.running)


if __name__ == "__main__":
    unittest.main()
