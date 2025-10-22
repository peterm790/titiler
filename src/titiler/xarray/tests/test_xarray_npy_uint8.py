import numpy
from io import BytesIO

import pytest

from starlette.testclient import TestClient

from titiler.xarray.examples.local import app


client = TestClient(app)


def test_npy_uint8_with_rescale():
    r = client.get(
        "/tiles/WebMercatorQuad/0/0/0.npy",
        params={
            "npy_uint8": True,
            "rescale": "-1,1",
        },
    )
    assert r.status_code == 200
    arr = numpy.load(BytesIO(r.content))
    # last band is mask
    assert arr.ndim == 3
    assert arr.dtype == numpy.uint8


def test_npy_uint8_missing_rescale_errors_for_non_uint8():
    r = client.get(
        "/tiles/WebMercatorQuad/0/0/0.npy",
        params={
            "npy_uint8": True,
        },
    )
    assert r.status_code == 400

