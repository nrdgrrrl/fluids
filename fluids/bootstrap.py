"""Shared Vulkan backend policy and native window creation."""

import os
import sys

import taichi as ti


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


def create_window() -> ti.ui.Window:
    """Create Taichi's resizable native window with vertical sync enabled."""
    # The Linux wheel uses GLFW/X11. Avoid its native abort in a headless session;
    # Wayland desktops need XWayland and its DISPLAY environment variable.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "Native window initialization failed. Run Fluids in a desktop graphics "
            "session with DISPLAY set (enable XWayland on Wayland desktops)."
        )
    try:
        # Taichi's GGUI window is resizable by default; no resizable keyword exists.
        return ti.ui.Window("Fluids — coastal wave sandbox", (1440, 900), vsync=True)
    except Exception as exc:
        raise RuntimeError(
            "Native window initialization failed. Install a compatible Vulkan GPU "
            "driver and run Fluids in a desktop graphics session with Taichi GGUI support."
        ) from exc
