import abc
from abc import ABC
from functools import partial
from typing import Any

from langchain_core.documents.base import Document
from PIL.Image import Image
from pydantic import BaseModel

from models.listings import Listing


class AbstractVectorDBManager(ABC):
    models = [
        ("listings", Listing),
    ]

    def __init__(self) -> None:
        self._db_connection = None

    def init(self, reset: bool = False) -> None:
        self._init_db(reset)
        self._init_models(reset)

    @abc.abstractmethod
    def _init_db(self, reset: bool = False) -> None:
        raise NotImplementedError()

    def _init_models(self, reset: bool = False) -> None:

        def _build_default_meth(method_name):
            def _no_implemented(
                self, model_object: BaseModel, model_name: str, reset: bool
            ) -> None:
                raise NotImplementedError(
                    f"{method_name}(self, model_object:BaseModel, model_name:str, reset:bool)"
                )

            return partial(_no_implemented, self=self)

        for model_name, model_object in self.models:
            # execute _init_{model_name}
            init_method_name = f"_init_{model_name}"
            init_method = getattr(
                self, init_method_name, _build_default_meth(init_method_name)
            )
            init_method(model_object=model_object, model_name=model_name, reset=reset)

            if reset or self._is_table_empty(model_name):
                # execute _load_{model_name}_data
                load_method_name = f"_load_{model_name}_data"
                load_method = getattr(
                    self, load_method_name, _build_default_meth(load_method_name)
                )
                load_method(model_object, model_name, reset)

    @abc.abstractmethod
    def _is_table_empty(self, model_name: str) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def _text_image_search(self, text: str, image: Image, limit: int = 3) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def _text_search(self, text: str) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def _image_search(self, image: Image, limit: int = 3) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_by_id(self, id: str) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def _retrieve_documents(
        self,
        query_result: Any,
        columns: list[str] | None = None,
        text_field: str = None,
        limit: int = 3,
    ) -> Document:
        raise NotImplementedError()
