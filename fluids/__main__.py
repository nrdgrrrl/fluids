"""Console and python -m entry point for the initial empty-window checkpoint."""

import sys


def main() -> int:
    try:
        from .bootstrap import create_window, initialize_gpu
    except (ImportError, OSError) as exc:
        print(
            "fluids: Unable to load Taichi. Reinstall the package in a Python 3.11 "
            f"virtual environment using 'python -m pip install .'. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        initialize_gpu()
        window = create_window()
        try:
            canvas = window.get_canvas()
            while window.running:
                canvas.set_background_color((0.035, 0.055, 0.085))
                window.show()
        finally:
            window.destroy()
    except RuntimeError as exc:
        print(f"fluids: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"Details: {exc.__cause__}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
