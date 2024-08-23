import abc
import logging
import os
import shutil
import uuid
from abc import ABC
from functools import partial
from typing import Any

import lancedb
import pandas as pd
import PIL
import torch
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from lancedb.table import LanceQueryBuilder
from langchain_core.documents.base import Document
from PIL.Image import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

from config import CONFIG
from models.listings import Listing, get_listing_summary
from utils import embedd_image, embedd_text, pil_to_bytes, singleton

_logger = logging.getLogger(__name__)


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
                load_method(
                    model_object=model_object, model_name=model_name, reset=reset
                )

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


@singleton()
class LanceDBManager(AbstractVectorDBManager):
    _table_name: str = "listings"
    _text_vector_column: str = "vector"
    _image_vector_column: str = "image_vector"

    def __init__(self) -> None:
        super().__init__()

    def _init_db(self, reset: bool) -> None:
        if reset and os.path.exists(CONFIG.VECTOR_DB_URI):
            shutil.rmtree(CONFIG.VECTOR_DB_URI)
        self._db_connection = lancedb.connect(CONFIG.VECTOR_DB_URI)

    def _is_table_empty(self, model_name: str) -> bool:
        return not self._get_table(model_name).count_rows()

    def _get_table(self, model_name: str) -> lancedb.table.Table | None:
        try:
            return self._db_connection.open_table(model_name)
        except Exception as e:
            _logger.exception(e)
            return None

    def _init_listings(
        self, model_object: BaseModel, model_name: str, reset: bool
    ) -> None:
        print(model_object, model_name, reset, self._get_table(model_name))
        if reset or not self._get_table(model_name):
            self._db_connection.create_table(
                model_name, schema=model_object.to_arrow_schema()
            )
        return

    def _load_listings_data(
        self, model_object: BaseModel, model_name: str, reset: bool
    ) -> None:
        table = self._db_connection.open_table(model_name)
        listing_file = CONFIG.LISTING_FILE
        if not os.path.exists(listing_file):
            raise Exception(f"Listings files non-existant: {listing_file}")

        def _load_listings(**kwargs):
            df = pd.read_csv(listing_file)
            for record in df.to_dict("records"):
                yield record

        clip_model_name = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        def _process_dataset(batch: LazyBatch) -> LazyBatch:
            keys = list(batch.keys_to_format)
            listing_summaries = [
                get_listing_summary(dict(zip(keys, data)))
                for data in zip(*[batch[k] for k in keys])
            ]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch["image"] = list(map(PIL.Image.open, batch["picture_file"]))
            image = clip_processor(
                text=None, images=batch["image"], return_tensors="pt"
            )["pixel_values"]
            image.to(device)
            batch["image"] = list(map(pil_to_bytes, batch["image"]))
            batch["image_vector"] = (
                clip_model.to(device).get_image_features(image).cpu()
            )
            batch["listing_summary"] = listing_summaries
            batch["vector"] = embedd_text(listing_summaries)
            return batch

        # trick to avoid caching the generator. more info bellow
        # https://discuss.huggingface.co/t/is-from-generator-caching-how-to-stop-it/70013/2
        gen_kwargs = dict(dummy_key=uuid.uuid4())
        listings_dataset = Dataset.from_generator(
            generator=_load_listings, gen_kwargs=gen_kwargs
        )
        processed_ds = listings_dataset.map(
            _process_dataset, batched=True, batch_size=10
        )

        table.add(
            list(map(lambda record: model_object(**record), processed_ds.to_list()))
        )
        _logger.info("Vector db sucessfully initialized")
        _logger.info("Listing table: %s record(s)", table.count_rows())

    def _text_image_search(
        self, text: str, image: Image, limit: int = 3
    ) -> LanceQueryBuilder:
        # first search for the ids of listings matching the text
        text_matching_ids = self._text_search(text).select(["id"]).limit(limit)
        # then search in the subset of listings those whose picture resemble most to input
        return self._image_search(image, limit=limit).where(
            f"id in f{text_matching_ids}", prefilter=True
        )

    def _text_search(self, text: str) -> LanceQueryBuilder:
        return self._get_table(self._table_name).search(
            query=embedd_text(text)[0],
            vector_column_name=self._text_vector_column,
        )

    def _image_search(self, image: Image) -> LanceQueryBuilder:
        return self._get_table(self._table_name).search(
            query=embedd_image(image)[0],
            vector_column_name=self._image_vector_column,
        )

    def _get_by_id(self, id: str) -> LanceQueryBuilder:
        return (
            self._get_table(self._table_name)
            .search()
            .where(f"id='{id}'", prefilter=True)
        )

    def _retrieve_documents(
        self,
        query_result: LanceQueryBuilder,
        columns: list[str] | None = None,
        text_field: str = None,
        limit: int = 3,
    ) -> Document:
        columns = columns or list(
            set(Listing.field_names()) - {"vector", "image_vector"}
        )
        if not text_field:
            text_field = "listing_summary"
        if text_field not in columns:
            columns.append(text_field)
        if "id" not in columns:
            columns.append("id")

        def _process_record(record):
            page_context = record.pop(text_field)
            return Document(page_content=page_context, metadata=record)

        return list(
            map(_process_record, query_result.select(columns).limit(limit).to_list())
        )


def get_vectordb_manager(engine: str) -> AbstractVectorDBManager:
    _manager_map = {
        "lancedb": LanceDBManager(),
        "chromadb": None,  # TODO:
    }
    return _manager_map[engine]
