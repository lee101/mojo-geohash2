import importlib.metadata
import importlib.util
import pathlib
import sys

import pytest


@pytest.fixture(scope="session")
def upstream_geohash2():
    """Load PyPI geohash2 without letting this checkout's package shadow it."""
    package_dir = pathlib.Path(
        importlib.metadata.distribution("geohash2").locate_file("geohash2")
    )
    spec = importlib.util.spec_from_file_location(
        "_upstream_geohash2", package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
