
from ..utils import singleton


def test_singleton():

    @singleton
    class DummyClass(object):
        def __init__(self) -> None:
            pass

    a = DummyClass()
    b = DummyClass()
    assert id(a) == id(b)
