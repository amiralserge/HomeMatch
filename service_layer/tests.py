import json
import os
import re
import shutil
from collections import defaultdict
from functools import partial
from typing import Any
from unittest import mock

import pytest
from langchain_core.documents.base import Document
from PIL.Image import Image
from pydantic import BaseModel

from utils.utils import singleton

from .vector_db_managers import AbstractVectorDBManager, LanceDBManager


class TestAbstractVectorDBManager:

    @classmethod
    def getVectorDBManagerStubClass(cls, model_names: list):
        class VectorDBManagerStub(AbstractVectorDBManager):
            models = [(model, mock.MagicMock()) for model in model_names]

            def _init_db(self, reset: bool = False) -> None:
                raise NotImplementedError

            def _is_table_empty(self, model_name: str) -> bool:
                raise NotImplementedError

            def _text_image_search(
                self, text: str, image: Image, limit: int = 3
            ) -> Any:
                raise NotImplementedError

            def _text_search(self, text: str) -> Any:
                raise NotImplementedError

            def _image_search(self, image: Image, limit: int = 3) -> Any:
                raise NotImplementedError

            def _get_by_id(self, id: str) -> Any:
                raise NotImplementedError

            def _retrieve_documents(
                self,
                query_result: mock,
                columns: list[str] | None = None,
                text_field: str = None,
                limit: int = 3,
            ) -> Document:
                raise NotImplementedError

        return VectorDBManagerStub

    def test_init_models(self):
        class Stub(self.getVectorDBManagerStubClass(model_names=["todo_task"])):
            pass

        manager = Stub()
        expected_error_msg = re.escape(
            "_init_todo_task(self, model_object:BaseModel, model_name:str, reset:bool)"
        )
        with pytest.raises(NotImplementedError, match=expected_error_msg):
            manager._init_models()

        class Stub2(Stub):
            def _init_todo_task(
                self, model_object: BaseModel, model_name: str, reset: bool
            ):
                pass

            def _is_table_empty(self, model_name: str) -> bool:
                return True

        manager = Stub2()
        expected_error_msg = re.escape(
            "_load_todo_task_data(self, model_object:BaseModel, model_name:str, reset:bool)"
        )
        with pytest.raises(NotImplementedError, match=expected_error_msg):
            manager._init_models()

    def test(self):

        @singleton(init_once=True)
        class FakeDB:
            def __init__(self) -> None:
                self._tables = defaultdict(list)

        class TodoTask(BaseModel):
            id: int
            name: str
            priority: int

        inital_db_data = []

        class FakeDBManager(self.getVectorDBManagerStubClass(model_names=[])):
            models = [("todo_task", TodoTask)]

            def _init_db(self, reset: bool = False) -> None:
                if reset:
                    FakeDB.instance = None
                self._db_connection = FakeDB()

            def _is_table_empty(self, model_name: str) -> bool:
                return len(self._db_connection._tables[model_name]) == 0

            def _init_todo_task(
                self, model_object: BaseModel, model_name: str, reset: bool
            ):  # noqa: E231
                if model_name not in self._db_connection._tables or reset:
                    self._db_connection._tables[model_name] = []

            def _load_todo_task_data(
                self, model_object: BaseModel, model_name: str, reset: bool
            ):  # noqa: E231
                if reset:
                    self._db_connection._tables[model_name].clear()

                self._db_connection._tables[model_name].extend(
                    [model_object(**data) for data in inital_db_data]
                )

        manager = FakeDBManager()
        assert manager._db_connection is None

        inital_db_data = [
            dict(id=1, name="Do the Dishes", priority=3),
            dict(id=2, name="Do the Laundry", priority=2),
        ]
        manager.init()
        assert manager._db_connection._tables["todo_task"] == [
            TodoTask(id=1, name="Do the Dishes", priority=3),
            TodoTask(id=2, name="Do the Laundry", priority=2),
        ]

        inital_db_data.extend(
            [
                dict(id=3, name="Cleanup the House", priority=5),
                dict(id=4, name="Cook Dinner", priority=5),
                dict(id=5, name="Iron Clothes", priority=3),
                dict(id=6, name="Pick Up Kids from school", priority=5),
            ]
        )

        manager.init()
        assert manager._db_connection._tables["todo_task"] == [
            TodoTask(id=1, name="Do the Dishes", priority=3),
            TodoTask(id=2, name="Do the Laundry", priority=2),
        ]

        manager.init(reset=True)
        assert manager._db_connection._tables["todo_task"] == [
            TodoTask(id=1, name="Do the Dishes", priority=3),
            TodoTask(id=2, name="Do the Laundry", priority=2),
            TodoTask(id=3, name="Cleanup the House", priority=5),
            TodoTask(id=4, name="Cook Dinner", priority=5),
            TodoTask(id=5, name="Iron Clothes", priority=3),
            TodoTask(id=6, name="Pick Up Kids from school", priority=5),
        ]


@mock.patch.object(LanceDBManager, "_load_listings_data")
class TestLanceDBManager:

    def test_singleton(self, mock_load_listing_data):
        LanceDBManager.instance = None
        manager1 = LanceDBManager()
        manager2 = LanceDBManager()
        assert manager1 is manager2

    @mock.patch("service_layer.vector_db_managers.CONFIG")
    def test(self, mock_config, mock_load_listing_data):
        LanceDBManager.instance = None

        sample_listing_data_file = "./service_layer/tests_data/sample_listing_data.json"
        if not os.path.exists(sample_listing_data_file):
            pytest.skip(f"Couldn't find `{sample_listing_data_file}`")

        if os.path.exists("./test.lance.db.manager"):
            try:
                shutil.rmtree("./test.lance.db.manager")
            except Exception:
                pytest.skip(
                    "Found './test.lance.db.manager' already existing and couldnt delete"
                )

        mock_config.base = "./test.lance.db.manager"
        mock_config.VECTOR_DB_URI = "./test.lance.db.manager"
        mock_config.VECTOR_DB_URI = "./test.lance.db.manager"

        with open(sample_listing_data_file, "r") as file:
            sample_data = json.load(file)

        def _load_listing_data(
            self, model_object: BaseModel, model_name: str, reset: bool
        ):
            table = self._db_connection.open_table(model_name)
            table.add([model_object(**data) for data in sample_data])

        manager = LanceDBManager()
        mock_load_listing_data.side_effect = partial(_load_listing_data, self=manager)
        manager.init()

        assert os.path.exists("./test.lance.db.manager")
        assert manager._is_table_empty("listings") is False
        assert manager._get_table("listings").count_rows() == len(sample_data)
        manager._db_connection.drop_database()
        manager._text_image_search()
