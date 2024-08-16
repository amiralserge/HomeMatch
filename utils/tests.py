
from ..utils import singleton, split_in_chunks


def test_singleton():

    @singleton
    class DummyClass(object):
        def __init__(self) -> None:
            pass

    a = DummyClass()
    b = DummyClass()
    assert id(a) == id(b)


def test_split_in_chunks():
    test_it = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    res = split_in_chunks(test_it, 3)
    assert res[0] == [1, 2, 3]
    assert res[1] == [4, 5, 6]
    assert res[2] == [7, 8, 9]
    assert res[3] == [10]
