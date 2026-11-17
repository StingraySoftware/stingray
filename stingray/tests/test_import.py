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


def _stingray_modules():
    """Names of all non-test modules of the package, e.g. ``stingray.io``."""
    import pkgutil

    import stingray

    return [
        m.name
        for m in pkgutil.walk_packages(stingray.__path__, "stingray.")
        if not m.ispkg
        and ".tests" not in m.name
        and m.name.rsplit(".", 1)[-1] not in ("conftest", "_version")
        and not m.name.rsplit(".", 1)[-1].startswith("test_")
    ]


@pytest.mark.parametrize("module", _stingray_modules())
def test_module_declares_existing_public_names(module):
    """Every module lists its public names in ``__all__``, and every listed name exists."""
    import importlib

    mod = importlib.import_module(module)
    assert hasattr(mod, "__all__")
    assert [name for name in mod.__all__ if not hasattr(mod, name)] == []


def test_top_level_names_exist():
    """Every name in ``stingray.__all__`` can be accessed from ``stingray``."""
    import stingray

    assert [name for name in stingray.__all__ if not hasattr(stingray, name)] == []
