"""Shared Vulkan backend policy and native window creation."""

import os
import sys

import glfw
import taichi as ti
import wgpu


VULKAN_ERROR = (
    "Vulkan initialization failed. Install a compatible GPU driver "
    "and run this application in a desktop graphics session."
)


def initialize_gpu() -> None:
    """Initialize Vulkan explicitly, refusing CPU fallback and backend overrides."""
    try:
        # Taichi honors TI_ARCH even when an explicit arch is passed to init.
        if os.environ.get("TI_ARCH", "vulkan") != "vulkan":
            raise RuntimeError("Unset TI_ARCH or set TI_ARCH=vulkan; Fluids requires Vulkan.")
        ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=True)
        if ti.cfg.arch != ti.vulkan:
            ti.reset()
            raise RuntimeError("Taichi selected a non-Vulkan backend; Fluids requires Vulkan.")
    except Exception as exc:
        raise RuntimeError(VULKAN_ERROR) from exc


class VulkanWindow:
    """Resizable GLFW window presenting the checkpoint's clear color via Vulkan."""

    def __init__(self) -> None:
        self._window = None
        self._context = None
        self._device = None
        self._background = (0.035, 0.055, 0.085, 1.0)
        if sys.platform.startswith("linux"):
            # Match the supported X11/XWayland desktop path.
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the desktop display.")
        try:
            glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self._window = glfw.create_window(
                1440, 900, "Fluids — coastal wave sandbox", None, None
            )
            if not self._window:
                raise RuntimeError("GLFW could not create a native window.")

            present_info = {"vsync": True}
            if sys.platform.startswith("linux"):
                present_info.update(
                    platform="x11", window=glfw.get_x11_window(self._window),
                    display=glfw.get_x11_display(),
                )
            elif sys.platform == "win32":
                present_info["window"] = glfw.get_win32_window(self._window)
            elif sys.platform == "darwin":
                present_info["window"] = glfw.get_cocoa_window(self._window)
            else:
                raise RuntimeError("Unsupported native window platform.")
            self._context = wgpu.gpu.get_canvas_context(present_info)

            # wgpu-native exposes backend selection through this environment
            # variable. Restore it after the request and verify the result too.
            backend = os.environ.get("WGPU_BACKEND_TYPE")
            if backend not in (None, "Vulkan"):
                raise RuntimeError("Unset WGPU_BACKEND_TYPE or set it to Vulkan.")
            os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"
            try:
                adapter = wgpu.gpu.request_adapter_sync(
                    canvas=self._context, power_preference="high-performance",
                    force_fallback_adapter=False,
                )
            finally:
                if backend is None:
                    os.environ.pop("WGPU_BACKEND_TYPE", None)
                else:
                    os.environ["WGPU_BACKEND_TYPE"] = backend
            if adapter.info["backend_type"] != "Vulkan":
                raise RuntimeError("The native renderer requires Vulkan.")
            self._device = adapter.request_device_sync()
            self._context.set_physical_size(*glfw.get_framebuffer_size(self._window))
            self._context.configure(
                device=self._device,
                format=self._context.get_preferred_format(adapter),
            )
            glfw.show_window(self._window)
        except BaseException:
            self.destroy()
            raise

    @property
    def running(self) -> bool:
        glfw.poll_events()
        return bool(self._window) and not glfw.window_should_close(self._window)

    def get_canvas(self):
        return self

    def set_background_color(self, color) -> None:
        self._background = (*color, 1.0)

    def show(self) -> None:
        width, height = glfw.get_framebuffer_size(self._window)
        if width == 0 or height == 0:
            # Minimized windows have no drawable surface. Keep processing close
            # and restore events without attempting a zero-sized swapchain.
            glfw.wait_events_timeout(0.05)
            return
        # The context recreates its swapchain when the physical size changes.
        self._context.set_physical_size(width, height)
        texture = self._context.get_current_texture()
        encoder = self._device.create_command_encoder()
        render_pass = encoder.begin_render_pass(color_attachments=[{
            "view": texture.create_view(), "resolve_target": None,
            "clear_value": self._background, "load_op": "clear", "store_op": "store",
        }])
        render_pass.end()
        self._device.queue.submit([encoder.finish()])
        self._context.present()

    def destroy(self) -> None:
        try:
            if self._context is not None:
                self._context.unconfigure()
                self._context = None
            if self._device is not None:
                self._device.destroy()
                self._device = None
        finally:
            if self._window:
                glfw.destroy_window(self._window)
                self._window = None
            glfw.terminate()


def create_window() -> VulkanWindow:
    """Create a resizable native window with Vulkan rendering and vertical sync."""
    # The Linux wheel uses GLFW/X11. Avoid its native abort in a headless session;
    # Wayland desktops need XWayland and its DISPLAY environment variable.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "Native window initialization failed. Run Fluids in a desktop graphics "
            "session with DISPLAY set (enable XWayland on Wayland desktops)."
        )
    try:
        # Taichi 1.7.4's GGUI fixes the window size. Use GLFW's explicit resize
        # support and a Vulkan-only presentation surface instead.
        return VulkanWindow()
    except Exception as exc:
        raise RuntimeError(
            "Native window initialization failed. Install a compatible Vulkan GPU "
            "driver and run Fluids in a desktop graphics session with GLFW support."
        ) from exc
