"""Run the command line as ``python -m admetus``.

The ``admetus`` console script is the documented entry point. This module exists so
a caller that knows only an interpreter path -- a subprocess, a container entry
point, a measurement script -- can reach the same command without having to locate
the script in a virtual environment's ``bin`` directory.
"""

from __future__ import annotations

from admetus.cli.main import main

if __name__ == "__main__":
    main()
