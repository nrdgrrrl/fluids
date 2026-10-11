"""A native GPU coastal-wave sandbox.

Importing this package must stay free of side effects: GPU initialization
happens in the entry point so that configuration and geometry can be imported
by tests without opening a window or claiming a device.
"""

__version__ = '0.1.0'
