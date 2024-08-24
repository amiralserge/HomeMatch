import logging
from functools import partial
from typing import List

from langchain_core.documents.base import Document
from PIL.Image import Image

from config import CONFIG
from utils.utils import singleton

from .constants import DEFAULT_VECTOR_DB_ENGINE
from .vector_db_managers import get_vectordb_manager

_logger = logging.getLogger(__name__)


@singleton(init_once=True)
class ListingsService(object):

    class InvalidSearchArgsException(Exception):
        pass

    def __init__(self, engine=None) -> None:
        engine = engine or CONFIG.vector_db_engine or DEFAULT_VECTOR_DB_ENGINE
        self._db_manager = get_vectordb_manager(engine=engine)
        self._db_manager.init()

    def search(
        self,
        text: str = None,
        image: Image = None,
        text_field: str = None,
        limit: int = 3,
        columns: List[str] | None = None,
    ) -> list[Document]:
        if not (text or image):
            raise self.__class__.InvalidSearchArgsException(
                "Invalid arguments: at least one of text and image must be provided"
            )
        retrieve_fn = partial(
            self._db_manager._retrieve_documents,
            columns=columns,
            text_field=text_field,
            limit=limit,
        )
        if text and image:
            query = self._db_manager._text_image_search(text, image)
        if text:
            query = self._db_manager._text_search(text)
        if image:
            query = self._db_manager._image_search(image)
        return retrieve_fn(query_result=query)

    def get_by_id(
        self,
        id: str,
        columns: list[str] | None = None,
        text_field: str = None,
    ) -> list[Document]:
        return self._db_manager._retrieve_documents(
            self._db_manager._get_by_id(id),
            columns=columns,
            text_field=text_field,
            limit=1,
        )


def get_relevant_listings(
    text: str = None,
    image: Image = None,
    columns: list[str] | None = None,
    text_field: str = None,
    limit: int = 3,
) -> List[Document]:
    return ListingsService().search(
        text=text, image=image, columns=columns, text_field=text_field, limit=limit
    )


def get_listing_by_id(
    id: str, columns: list[str] | None = None, text_field: str = None
) -> Document:
    return ListingsService().get_by_id(id=id, columns=columns, text_field=text_field)
