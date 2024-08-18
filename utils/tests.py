from unittest import mock

import pytest

from .utils import local_image_to_data_url, singleton, split_in_chunks


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


@mock.patch("utils.utils.guess_type")
@mock.patch("utils.utils.b64encode_image")
@mock.patch("utils.utils.open_image")
def test_local_image_to_data_url(open_image_mock, encode_image_mock, guess_type_mock):
    guess_type_mock.return_value = (None, None)
    with pytest.raises(Exception, match="Could not detect mime type of file `test.img`"):
        local_image_to_data_url("test.img")

    guess_type_mock.return_value = ("image/jpeg", None)
    encode_image_mock.return_value = "oTQjRXWoXj9Sw9+RZQSs+2mrNSIXRApUWgcTcx0iARWYl5gdYQyXzwdXra35Wd0AShs="
    expected_result = "data:image/jpeg;base64,oTQjRXWoXj9Sw9+RZQSs+2mrNSIXRApUWgcTcx0iARWYl5gdYQyXzwdXra35Wd0AShs="
    assert local_image_to_data_url("test.img") == expected_result