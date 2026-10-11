"""Console and python -m entry point for the desktop simulation."""

import sys


def main() -> int:
    try:
        from .app import run
    except (ImportError, OSError) as exc:
        print(
            "fluids: Unable to load Taichi. Reinstall the package in a Python 3.11 "
            f"virtual environment using 'python -m pip install .'. Details: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        run()
    except (OSError, RuntimeError) as exc:
        print(f"fluids: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"Details: {exc.__cause__}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
