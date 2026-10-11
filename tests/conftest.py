"""Shared fixtures.

The bootstrap tests cover this package's backend policy and error handling,
not Taichi itself, so they run against a stub module. That keeps them usable
on machines without a GPU; the real Vulkan path can only be verified by
running the application on target hardware.
"""

import importlib
import sys
import types

import pytest

_MODULES_UNDER_TEST = ('fluids.bootstrap', 'fluids.__main__')


class FakeWindow:
    """Minimal stand-in for ``ti.ui.Window`` that closes after a few frames."""

    def __init__(self, stub, title, resolution, vsync=False):
        if stub.window_error is not None:
            raise stub.window_error
        self.title = title
        self.resolution = resolution
        self.vsync = vsync
        self.frames = 0
        self.max_frames = 3
        self.background_colors = []
        stub.windows.append(self)

    @property
    def running(self):
        return self.frames < self.max_frames

    def get_canvas(self):
        return self

    def set_background_color(self, color):
        self.background_colors.append(color)

    def show(self):
        self.frames += 1


def _build_stub():
    taichi = types.ModuleType('taichi')
    lang = types.ModuleType('taichi.lang')
    impl = types.ModuleType('taichi.lang.impl')

    taichi.vulkan = 'arch:vulkan'
    taichi.cpu = 'arch:cpu'
    taichi.init_calls = []
    taichi.init_error = None
    taichi.window_error = None
    taichi.active_arch = taichi.vulkan
    taichi.windows = []

    def init(**kwargs):
        taichi.init_calls.append(kwargs)
        if taichi.init_error is not None:
            raise taichi.init_error

    taichi.init = init
    taichi.ui = types.SimpleNamespace(
        Window=lambda title, resolution, vsync=False: FakeWindow(
            taichi, title, resolution, vsync
        )
    )

    impl.current_cfg = lambda: types.SimpleNamespace(arch=taichi.active_arch)
    lang.impl = impl
    taichi.lang = lang
    return taichi, lang, impl


@pytest.fixture
def taichi_stub(monkeypatch):
    taichi, lang, impl = _build_stub()
    monkeypatch.setitem(sys.modules, 'taichi', taichi)
    monkeypatch.setitem(sys.modules, 'taichi.lang', lang)
    monkeypatch.setitem(sys.modules, 'taichi.lang.impl', impl)
    for name in _MODULES_UNDER_TEST:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield taichi
    # Later tests must not reuse modules bound to this stub.
    for name in _MODULES_UNDER_TEST:
        sys.modules.pop(name, None)


@pytest.fixture
def load(taichi_stub):
    """Import a module under test fresh, bound to the Taichi stub."""

    def _load(name):
        sys.modules.pop(name, None)
        return importlib.import_module(name)

    return _load
