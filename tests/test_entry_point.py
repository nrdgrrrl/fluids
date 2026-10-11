"""Entry point behaviour: run until the window closes, or report why not."""

import subprocess
import sys


def test_window_loop_runs_until_window_closes(taichi_stub, load):
    entry = load('fluids.__main__')

    assert entry.main([]) == 0

    window = taichi_stub.windows[-1]
    assert window.frames == window.max_frames
    assert window.background_colors[0] == entry.BACKGROUND_COLOR


def test_check_gpu_initializes_without_opening_a_window(taichi_stub, load, capsys):
    entry = load('fluids.__main__')

    assert entry.main(['--check-gpu']) == 0

    assert taichi_stub.init_calls
    assert taichi_stub.windows == []
    assert 'Vulkan' in capsys.readouterr().out


def test_failed_backend_reports_error_and_exit_code(taichi_stub, load, capsys):
    entry = load('fluids.__main__')
    taichi_stub.init_error = RuntimeError('no vulkan device')

    assert entry.main([]) == 1

    captured = capsys.readouterr()
    assert 'Vulkan initialization failed' in captured.err
    assert taichi_stub.windows == []


def test_importing_the_package_does_not_touch_the_gpu():
    # The package must stay importable, and import-time silent, on machines
    # where Taichi is not installed at all.
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys, fluids; assert "taichi" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
