from typing import Any, List, Union

import torch
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor

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
