"""Desktop entry point.

This is a runnable checkpoint before the simulation modules exist; a later
issue replaces the empty loop with the complete application.
"""

import argparse
import sys

from .bootstrap import create_window, initialize_gpu

BACKGROUND_COLOR = (0.035, 0.055, 0.085)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='fluids', description='A native GPU coastal-wave sandbox.'
    )
    parser.add_argument(
        '--check-gpu',
        action='store_true',
        help='Initialize the Vulkan backend, report the result, and exit '
        'without opening a window.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        initialize_gpu()
        if args.check_gpu:
            # Verifying the installation on target hardware should not require
            # a window or a frame of rendering.
            print('Vulkan backend initialized.')
            return 0
        window = create_window()
    except RuntimeError as exc:
        # The bootstrap failure text is the actionable message; a traceback
        # here would bury it.
        print(f'fluids: {exc}', file=sys.stderr)
        return 1

    canvas = window.get_canvas()
    # Keep the first checkpoint runnable before simulation modules are added.
    while window.running:
        canvas.set_background_color(BACKGROUND_COLOR)
        window.show()
    # Closing the window ends the loop, and the process exits with it.
    return 0


if __name__ == '__main__':
    sys.exit(main())
