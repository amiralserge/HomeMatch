from unittest import mock

import pytest
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings

from .utils import (
    ClipImageEmbedding,
    NoEmbedderForDocumentTypeException,
    embedd_image,
    embedd_text,
    get_embedder,
    local_image_to_data_url,
    singleton,
    split_in_chunks,
)


class TestSigleton:

    def test_singleton(self):

        @singleton()
        class DummySingletonClass(object):
            def __init__(self) -> None:
                self.value = 0

        instance1 = DummySingletonClass()
        instance2 = DummySingletonClass()
        assert id(instance1) == id(instance2)

        instance1.value = 117
        assert instance2.value == 117
        assert DummySingletonClass().value == instance1.value == instance2.value == 0

    def test_init_once(self):

        @singleton(init_once=True)
        class DummySingletonClass(object):
            def __init__(self) -> None:
                self.value = 0

        instance1 = DummySingletonClass()
        instance2 = DummySingletonClass()
        assert id(instance1) == id(instance2)
        assert instance1.value == instance2.value == 0

        instance2.value = 1117
        assert instance1.value == 1117

        instance3 = DummySingletonClass()
        instance3.value == instance1.value == instance2.value == 1117


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
    with pytest.raises(
        Exception, match="Could not detect mime type of file `test.img`"
    ):
        local_image_to_data_url("test.img")

    guess_type_mock.return_value = ("image/jpeg", None)
    encode_image_mock.return_value = (
        "oTQjRXWoXj9Sw9+RZQSs+2mrNSIXRApUWgcTcx0iARWYl5gdYQyXzwdXra35Wd0AShs="
    )
    expected_result = "data:image/jpeg;base64,oTQjRXWoXj9Sw9+RZQSs+2mrNSIXRApUWgcTcx0iARWYl5gdYQyXzwdXra35Wd0AShs="
    assert local_image_to_data_url("test.img") == expected_result


def test_get_embedder():
    # test unknown type
    with pytest.raises(
        NoEmbedderForDocumentTypeException,
        match="No embedder found for document type: unknown",
    ):
        get_embedder(document_type="unknown")

    # test text
    embedder = get_embedder(document_type="text")
    assert type(get_embedder(document_type="text")) is OpenAIEmbeddings

    embedder = get_embedder(document_type="text", use_cache=True)
    assert type(embedder) is CacheBackedEmbeddings
    assert type(embedder.underlying_embeddings) is OpenAIEmbeddings

    # test image
    embedder = get_embedder(document_type="image")
    assert type(embedder) is ClipImageEmbedding

    embedder = get_embedder(document_type="image", use_cache=True)
    assert type(embedder) is CacheBackedEmbeddings
    assert type(embedder.underlying_embeddings) is ClipImageEmbedding


@mock.patch("utils.utils.__openai_text_embedder")
@mock.patch("utils.utils.__cached_openai_text_embedder")
def test_embedd_text(mock__cached_openai_text_embedder, mock___openai_text_embedder):
    embedd_text("some text input")
    mock___openai_text_embedder.embed_documents.assert_called_once_with(
        ["some text input"]
    )
    mock__cached_openai_text_embedder.embed_documents.assert_not_called()

    mock___openai_text_embedder.reset_mock()
    mock__cached_openai_text_embedder.reset_mock()

    embedd_text(documents=["some text input"], use_cache=True)
    mock___openai_text_embedder.embed_documents.assert_not_called()
    mock__cached_openai_text_embedder.embed_documents.assert_called_once_with(
        ["some text input"]
    )


@mock.patch("utils.utils.__clip_image_embedder")
@mock.patch("utils.utils.__cached_clip_image_embedder")
def test_embedd_image(mock____cached_clip_image_embedder, mock___clip_image_embedder):

    image = mock.Mock()
    embedd_image(image)
    mock___clip_image_embedder.embed_documents.assert_called_once_with([image])
    mock____cached_clip_image_embedder.embed_documents.assert_not_called()

    mock___clip_image_embedder.reset_mock()
    mock____cached_clip_image_embedder.reset_mock()

    embedd_image(documents=[image], use_cache=True)
    mock___clip_image_embedder.embed_documents.assert_not_called()
    mock____cached_clip_image_embedder.embed_documents.assert_called_once_with([image])
