"""Make the repository root importable for the test suite.

pytest prepends the *test file's* directory to sys.path, not the rootdir, so
``import LST_AI`` fails when the package has not been pip-installed. Inserting the
root here lets the tests run against a working tree with no install step -- which is
what keeps the CI architecture job independent of the heavy extras.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
