from .utils import singleton, split_in_chunks


def test_singleton():

    @singleton
    class DummyClass(object):
        def __init__(self) -> None:
            self.value = 0

    instance1 = DummyClass()
    instance2 = DummyClass()
    assert id(instance1) == id(instance2)

    instance1.value = 117
    assert instance2.value == 117
    assert DummyClass().value == 117


def test_split_in_chunks():
    test_it = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert split_in_chunks(test_it, 3) == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    assert split_in_chunks([], 3) == []
    assert split_in_chunks([1, 2], 3) == [[1, 2]]
    assert split_in_chunks([1, 2, 3], 5) == [[1, 2, 3]]
    assert split_in_chunks([1, 2, 3], 1) == [[1], [2], [3]]
