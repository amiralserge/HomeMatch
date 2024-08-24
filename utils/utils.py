import base64
import functools
import io
from io import BytesIO
from mimetypes import guess_type
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from PIL import Image as ImageModule
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor

IMG_SIZE = (640, 427)


def open_image(image_path) -> Image:
    return ImageModule.open(image_path)


def resize_image(image: Image, size=IMG_SIZE) -> Image:
    return image.resize(size)


def b64encode_image(image: Image, format: str) -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def local_image_to_data_url(image_path, size: Tuple = IMG_SIZE) -> str:
    """
    Function to encode a local image into data URL
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        # mime_type = 'image/png'
        raise Exception(f"Could not detect mime type of file `{image_path}`")
    format = mime_type.split("/")[-1]

    image = open_image(image_path)
    if size:
        image = resize_image(image, size=size)
    # Construct the data URL
    return f"data:{mime_type};base64,{b64encode_image(image, format=format)}"


def pil_to_bytes(image: Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="jpeg")
    return buffer.getvalue()


def split_in_chunks(iterator: List, chunk_size: int) -> list[List]:
    if not len(iterator):
        return []
    indices = np.arange(chunk_size, len(iterator), chunk_size)
    return list(map(lambda i: i.tolist(), np.array_split(iterator, indices)))


__text_embedding_store = LocalFileStore("./embedding_cache/text/")
__openai_text_embedder = OpenAIEmbeddings()
__cached_openai_text_embedder = CacheBackedEmbeddings.from_bytes_store(
    __openai_text_embedder,
    __text_embedding_store,
    namespace=__openai_text_embedder.model,
)

__clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
__clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True
)


class ClipImageEmbedding(Embeddings):
    model: str = "clip-vit-base-patch32"

    def __init__(self, device="cpu"):
        self._device = device

    def embed_documents(self, images: List[Image]) -> List[List[float]]:
        return self._embedd_images(images)

    def embed_query(self, image: Image) -> List[float]:
        return self._embedd_images(image)[0]

    def _embedd_images(self, images: Union[Image | List[Image]]) -> List[List[float]]:
        with torch.no_grad():
            inputs = __clip_processor.to(self._device)(images, return_tensors="pt")
            return list(
                map(
                    func=lambda e: e.tolist(),
                    iter1=__clip_model.to(self._device)
                    .get_image_features(**inputs)
                    .to("cpu"),
                )
            )


__image_embedding_store = LocalFileStore("./embedding_cache/image/")
__clip_image_embedder = ClipImageEmbedding()
__cached_clip_image_embedder = CacheBackedEmbeddings.from_bytes_store(
    __clip_image_embedder,
    __image_embedding_store,
    namespace=__clip_image_embedder.model,
)


def embedd_text(
    documents: Union[List[str] | str], use_cache=False
) -> List[List[float]]:
    return __embedd(document_type="text", documents=documents, use_cache=use_cache)


def embedd_image(
    documents: Union[List[Image] | Image], use_cache=False
) -> List[List[float]]:
    return __embedd(document_type="image", documents=documents, use_cache=use_cache)


def __embedd(
    document_type: str, documents: Union[List[Any] | Any], use_cache=False
) -> List[List[float]]:
    documents = documents if isinstance(documents, (list, tuple)) else [documents]
    embedder = get_embedder(document_type, use_cache=use_cache)
    return embedder.embed_documents(documents)


class NoEmbedderForDocumentTypeException(Exception):
    def __init__(self, document_type):
        self.document_type = document_type
        super().__init__(f"No embedder found for document type: {document_type}")

    pass


def get_embedder(document_type: str, use_cache: bool = False) -> Embeddings:
    if document_type == "text":
        return __cached_openai_text_embedder if use_cache else __openai_text_embedder
    if document_type == "image":
        return __cached_clip_image_embedder if use_cache else __clip_image_embedder
    raise NoEmbedderForDocumentTypeException(document_type=document_type)


def singleton(init_once: bool = False):
    """
    Decorator for creating singleton classes.

    :param init_once: If True, __init__ is called only once.
                      If False, __init__ is called on every instance creation.
    """
    def inner(klass):
        original__init__ = klass.__init__
        original__new__ = klass.__new__
        klass._instance = None
        klass._instance_initialized = False

        @functools.wraps(original__init__)
        def __init__(self, *args, **kwargs):
            if init_once and self.__class__._instance_initialized:
                return
            original__init__(self, *args, **kwargs)
            self.__class__._instance_initialized = True

        @functools.wraps(original__new__)
        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super(klass, cls).__new__(cls, *args, **kwargs)
            return cls._instance

        klass.__new__ = __new__
        klass.__init__ = __init__
        return klass

    return inner
