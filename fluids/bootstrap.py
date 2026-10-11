"""GPU backend and window bootstrap.

Backend policy lives here, apart from the entry point, so the eventual
application and the ``fluids --check-gpu`` smoke check initialize the device
the same way.
"""

import taichi as ti

WINDOW_TITLE = 'Fluids — coastal wave sandbox'
WINDOW_RESOLUTION = (1440, 900)

VULKAN_ERROR = (
    'Vulkan initialization failed. Install a compatible GPU driver '
    'and run this application in a desktop graphics session.'
)

WINDOW_ERROR = (
    'Could not open the application window. Run this application in a '
    'desktop graphics session with a Vulkan-capable display driver.'
)


def _active_arch():
    """Return the backend Taichi actually selected, or ``None`` if unknown."""
    try:
        from taichi.lang import impl

        return impl.current_cfg().arch
    except Exception:
        # The configuration accessor is internal; an unreadable value must not
        # by itself fail a working Vulkan initialization.
        return None


def initialize_gpu():
    """Initialize Taichi on Vulkan, raising rather than falling back to CPU."""
    try:
        # A CPU fallback would conceal an unsupported installation and violate
        # the performance architecture, so initialization must fail explicitly.
        ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=True)
    except Exception as exc:
        raise RuntimeError(VULKAN_ERROR) from exc

    arch = _active_arch()
    if arch is not None and arch != ti.vulkan:
        # Defends the same policy if a future Taichi release stops raising.
        raise RuntimeError(
            f'{VULKAN_ERROR} Taichi selected the {arch} backend instead.'
        )


def create_window():
    """Open the resizable native window used to present the simulation."""
    try:
        # Vsync limits unnecessary rendering work while physics has its own clock.
        return ti.ui.Window(WINDOW_TITLE, WINDOW_RESOLUTION, vsync=True)
    except Exception as exc:
        raise RuntimeError(WINDOW_ERROR) from exc
