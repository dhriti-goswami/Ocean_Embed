import pytest

from oceanembed.data.synthetic import make


@pytest.fixture(scope="session")
def cube(tmp_path_factory):
    """Synthetic cube with the real file's variable names and shapes (92 x 69 x 81, 35 depths)."""
    return make(str(tmp_path_factory.mktemp("data") / "cube.nc"))


@pytest.fixture(scope="session")
def cube_salinity(tmp_path_factory):
    return make(str(tmp_path_factory.mktemp("data_s") / "cube_s.nc"), with_salinity=True)
