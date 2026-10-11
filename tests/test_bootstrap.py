"""Backend policy: Vulkan or an actionable failure, never a quiet CPU run."""

import pytest


def test_init_requests_vulkan_without_fallback(taichi_stub, load):
    bootstrap = load('fluids.bootstrap')

    bootstrap.initialize_gpu()

    assert taichi_stub.init_calls == [
        {
            'arch': taichi_stub.vulkan,
            'enable_fallback': False,
            'offline_cache': True,
        }
    ]


def test_failed_init_raises_actionable_error(taichi_stub, load):
    bootstrap = load('fluids.bootstrap')
    taichi_stub.init_error = RuntimeError('no vulkan device')

    with pytest.raises(RuntimeError) as excinfo:
        bootstrap.initialize_gpu()

    assert str(excinfo.value).startswith(bootstrap.VULKAN_ERROR)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_silent_cpu_fallback_is_rejected(taichi_stub, load):
    bootstrap = load('fluids.bootstrap')
    # A Taichi release that stops raising must still not reach the CPU path.
    taichi_stub.active_arch = taichi_stub.cpu

    with pytest.raises(RuntimeError, match='Vulkan initialization failed'):
        bootstrap.initialize_gpu()


def test_unreadable_config_does_not_fail_a_working_init(taichi_stub, load):
    bootstrap = load('fluids.bootstrap')

    def exploding_cfg():
        raise AttributeError('internal API moved')

    taichi_stub.lang.impl.current_cfg = exploding_cfg

    bootstrap.initialize_gpu()


def test_create_window_uses_configured_title_and_resolution(taichi_stub, load):
    bootstrap = load('fluids.bootstrap')

    window = bootstrap.create_window()

    assert window.title == bootstrap.WINDOW_TITLE
    assert window.resolution == bootstrap.WINDOW_RESOLUTION
    assert window.vsync is True


def test_window_failure_raises_actionable_error(taichi_stub, load):
    bootstrap = load('fluids.bootstrap')
    taichi_stub.window_error = RuntimeError('no display')

    with pytest.raises(RuntimeError) as excinfo:
        bootstrap.create_window()

    assert str(excinfo.value) == bootstrap.WINDOW_ERROR
    assert isinstance(excinfo.value.__cause__, RuntimeError)
