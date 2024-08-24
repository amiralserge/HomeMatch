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

from .services import ListingsService
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

        class FakeDBManager(self.getVectorDBManagerStubClass(model_names=[])):
            models = [("todo_task", TodoTask)]
            inital_db_data = []

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
                    [model_object(**data) for data in self.inital_db_data]
                )

        manager = FakeDBManager()
        assert manager._db_connection is None

        FakeDBManager.inital_db_data = [
            dict(id=1, name="Do the Dishes", priority=3),
            dict(id=2, name="Do the Laundry", priority=2),
        ]
        manager.init()
        assert manager._db_connection._tables["todo_task"] == [
            TodoTask(id=1, name="Do the Dishes", priority=3),
            TodoTask(id=2, name="Do the Laundry", priority=2),
        ]

        # adding more items
        FakeDBManager.inital_db_data = [
            *FakeDBManager.inital_db_data,
            dict(id=3, name="Cleanup the House", priority=5),
            dict(id=4, name="Cook Dinner", priority=5),
            dict(id=5, name="Iron Clothes", priority=3),
            dict(id=6, name="Pick Up Kids from school", priority=5),
        ]

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

        manager = LanceDBManager()
        manager.init()
        assert manager._get_table("listings").count_rows() == len(sample_data)

        manager._db_connection.drop_database()


@mock.patch("service_layer.services.get_vectordb_manager")
class TestListingsService:

    @pytest.fixture
    @classmethod
    def setup(cls):
        ListingsService._instance = None
        ListingsService._instance_initialized = False
        yield
        ListingsService._instance = None
        ListingsService._instance_initialized = False

    @classmethod
    def getDummyVectorDBManagerClass(cls):
        class DummyVectorDBManager(AbstractVectorDBManager):
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
                query_result: Any,
                columns: list[str] | None = None,
                text_field: str = None,
                limit: int = 3,
            ) -> Document:
                raise NotImplementedError

        return DummyVectorDBManager

    def test_singleton(self, mock_get_vectordb_manager, setup):
        svc1 = ListingsService()
        svc2 = ListingsService()
        assert id(svc1) == id(svc2)

    def test_search(self, mock_get_vectordb_manager, setup):

        class DummyVectorDBManager(self.getDummyVectorDBManagerClass()):
            def init(self, reset: bool = False) -> None:
                pass

            def _text_search(self, text: str) -> Any:
                return [
                    dict(
                        description=(
                            "Welcome to Maple Grove, a charming abode designed for modern living. "
                            "This delightful 2-bedroom, 1-bathroom home's living area is tastefully decorated for relaxation. "
                            "The open-concept kitchen features modern appliances and a convenient bar stool area for casual meals. "
                            "A small yet inviting backyard provides a perfect spot for gardening or outdoor gatherings. "
                            "The house includes a single-car garage and a community clubhouse, creating a warm sense of belonging."
                        ),
                        neighborhood="Maple Grove",
                        city="Montreal",
                        province="Quebec",
                        address="402-117 Glory St",
                        price="$450,000",
                    ),
                    dict(
                        description=(
                            "Step into Birch Valley, where elegance meets comfort in this lovely 3-bedroom, 2-bathroom home. "
                            "The spacious living area features neutral-toned furniture and a warm ambiance, perfect for family movie nights. "
                            "The modern kitchen is designed for functionality and boasts stainless steel appliances and a sleek island for entertaining. "
                            "Relax in the peaceful backyard or utilize the community's fabulous amenities, including walking paths and a playground. "
                            "A two-car garage completes this wonderful home."
                        ),
                        neighborhood="Birch Valley",
                        city="Toronto",
                        province="Ontario",
                        address="708-1400 Valley St",
                        price="$700,000",
                    ),
                ]

            def _image_search(self, image: Image, limit: int = 3) -> Any:
                return [
                    dict(
                        description=(
                            "Step into Maple Grove, where tradition meets modernity in this charming 4-bedroom, 3-bathroom residence. "
                            "The open-concept living area invites natural light, with a cozy fireplace perfect for chilly evenings. "
                            "The gourmet kitchen is a chef's dream, featuring granite countertops and a large pantry. "
                            "Enjoy the beautifully landscaped backyard or explore the nearby park with its scenic trails and pond. "
                            "This home also includes a finished basement and a double garage."
                        ),
                        neighborhood="Maple Grove",
                        city="Vancouver",
                        province="British Columbia",
                        address="215-1600 Maple Dr",
                        price="$850,000",
                    ),
                    dict(
                        description=(
                            "Experience luxury living in Cedar Ridge with this stunning 5-bedroom, 4-bathroom home. "
                            "The grand foyer leads to an expansive living room with vaulted ceilings and a wall of windows. "
                            "The chef's kitchen is fully equipped with high-end appliances and a spacious breakfast nook. "
                            "Step outside to the private patio with an outdoor kitchen, perfect for summer BBQs. "
                            "This home features a home theater, gym, and a three-car garage."
                        ),
                        neighborhood="Cedar Ridge",
                        city="Montreal",
                        province="Quebec",
                        address="325-1800 Cedar Ln",
                        price="$1,200,000",
                    ),
                ]

            def _text_image_search(
                self, text: str, image: Image, limit: int = 3
            ) -> Any:
                return [
                    dict(
                        description=(
                            "Welcome to Pine Hill, a serene retreat with a 2-bedroom, 2-bathroom bungalow. "
                            "The cozy living room has a rustic charm with exposed wooden beams and a stone fireplace. "
                            "The eat-in kitchen is perfect for casual dining, featuring custom cabinetry and a farmhouse sink. "
                            "The expansive deck offers stunning views of the surrounding nature. "
                            "This property also includes a detached garage and a garden shed."
                        ),
                        neighborhood="Pine Hill",
                        city="Calgary",
                        province="Alberta",
                        address="412-1900 Pine St",
                        price="$650,000",
                    ),
                    dict(
                        description=(
                            "Discover urban living at its finest in Elmwood, with this sleek 2-bedroom, 2-bathroom condo. "
                            "The open-plan living and dining area is highlighted by floor-to-ceiling windows and modern finishes. "
                            "The contemporary kitchen boasts quartz countertops and stainless steel appliances. "
                            "Relax on the private balcony overlooking the city skyline. "
                            "This unit also includes a dedicated parking space and access to a rooftop terrace."
                        ),
                        neighborhood="Elmwood",
                        city="Toronto",
                        province="Ontario",
                        address="510-1450 Elmwood Ave",
                        price="$550,000",
                    ),
                ]

            def _retrieve_documents(
                self,
                query_result: Any,
                columns: list[str] | None = None,
                text_field: str = None,
                limit: int = 3,
            ) -> Document:
                return [
                    Document(
                        page_content=data[text_field],
                        metadata={k: data[k] for k in columns} if columns else data,
                    )
                    for data in query_result[:limit]
                ]

        dummy_vectordb_manager = DummyVectorDBManager()
        mock_get_vectordb_manager.return_value = dummy_vectordb_manager
        svc = ListingsService()
        mock_get_vectordb_manager.assert_called_once()
        assert svc._db_manager is dummy_vectordb_manager

        with pytest.raises(
            ListingsService.InvalidSearchArgsException,
            match=re.escape(
                "Invalid arguments: at least one of text and image must be provided"
            ),
        ):
            svc.search(text=None, image=None)

        fields = ["price", "city", "neighborhood", "address"]
        # text only search
        result = svc.search(
            text="Beautiful 2-bedrooms",
            columns=fields,
            text_field="description",
            limit=1,
        )
        expected_result = [
            Document(
                page_content=(
                    "Welcome to Maple Grove, a charming abode designed for modern living. "
                    "This delightful 2-bedroom, 1-bathroom home's living area is tastefully decorated for relaxation. "
                    "The open-concept kitchen features modern appliances and a convenient bar stool area for casual meals. "
                    "A small yet inviting backyard provides a perfect spot for gardening or outdoor gatherings. "
                    "The house includes a single-car garage and a community clubhouse, creating a warm sense of belonging."
                ),
                metadata=dict(
                    price="$450,000",
                    city="Montreal",
                    neighborhood="Maple Grove",
                    address="402-117 Glory St",
                ),
            )
        ]
        assert result == expected_result

        # image only search
        result = svc.search(
            image=mock.Mock(), columns=fields, text_field="description", limit=1
        )
        expected_result = [
            Document(
                page_content=(
                    "Step into Maple Grove, where tradition meets modernity in this charming 4-bedroom, 3-bathroom residence. "
                    "The open-concept living area invites natural light, with a cozy fireplace perfect for chilly evenings. "
                    "The gourmet kitchen is a chef's dream, featuring granite countertops and a large pantry. "
                    "Enjoy the beautifully landscaped backyard or explore the nearby park with its scenic trails and pond. "
                    "This home also includes a finished basement and a double garage."
                ),
                metadata=dict(
                    price="$850,000",
                    city="Vancouver",
                    neighborhood="Maple Grove",
                    address="215-1600 Maple Dr",
                ),
            ),
        ]
        assert result == expected_result

        # text and image search
        result = svc.search(
            text="Beautiful 2-bedrooms",
            image=mock.Mock(),
            columns=fields,
            text_field="description",
        )
        expected_result = [
            Document(
                page_content=(
                    "Welcome to Pine Hill, a serene retreat with a 2-bedroom, 2-bathroom bungalow. "
                    "The cozy living room has a rustic charm with exposed wooden beams and a stone fireplace. "
                    "The eat-in kitchen is perfect for casual dining, featuring custom cabinetry and a farmhouse sink. "
                    "The expansive deck offers stunning views of the surrounding nature. "
                    "This property also includes a detached garage and a garden shed."
                ),
                metadata=dict(
                    price="$650,000",
                    city="Calgary",
                    neighborhood="Pine Hill",
                    address="412-1900 Pine St",
                ),
            ),
            Document(
                page_content=(
                    "Discover urban living at its finest in Elmwood, with this sleek 2-bedroom, 2-bathroom condo. "
                    "The open-plan living and dining area is highlighted by floor-to-ceiling windows and modern finishes. "
                    "The contemporary kitchen boasts quartz countertops and stainless steel appliances. "
                    "Relax on the private balcony overlooking the city skyline. "
                    "This unit also includes a dedicated parking space and access to a rooftop terrace."
                ),
                metadata=dict(
                    price="$550,000",
                    city="Toronto",
                    neighborhood="Elmwood",
                    address="510-1450 Elmwood Ave",
                ),
            ),
        ]
        assert result == expected_result
