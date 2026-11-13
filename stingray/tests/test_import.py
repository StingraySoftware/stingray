import subprocess
import sys

import pytest

# Packages that only some functions need. ``import stingray`` must not load
# them, so that starting Stingray stays fast; they are imported inside the
# functions that use them.
HEAVY = [
    "matplotlib",
    "scipy.stats",
    "scipy.signal",
    "scipy.optimize",
    "astropy.table",
    "h5py",
]


@pytest.fixture(scope="module")
def loaded_by_import():
    """Packages in HEAVY that a fresh ``import stingray`` loads."""
    # A fresh interpreter, because in the test session they are already loaded.
    code = f"import sys, stingray; print(' '.join(p for p in {HEAVY!r} if p in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


@pytest.mark.parametrize("package", HEAVY)
def test_import_stingray_does_not_load(package, loaded_by_import):
    """``import stingray`` must not load a package that only some functions need."""
    assert package not in loaded_by_import
