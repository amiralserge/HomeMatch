from typing import Any

from dotenv import dotenv_values, load_dotenv

from utils import singleton

load_dotenv()


@singleton(init_once=True)
class config(object):

    def __init__(self) -> None:
        self.openai_base_url = "https://api.openai.com/v1"
        self.openai_api_key = None
        self.max_token = 2000
        self.llm_model = "gpt-4o-mini"
        self.llm_temperature = 0
        self.llm_request_cooldown_time = 5
        self.listing_pictures_dir = "./listing_pictures"
        self.listing_pictures_descr_file = (
            "./listing_pictures/pictures_descriptions.csv"
        )
        self.listing_file = "./picture_augmented_listings.csv"

        self.vector_db_engine = "lancedb"
        self.vector_db_uri = "./homematch"

        for key, value in dotenv_values().items():
            setattr(self, key.lower(), value)
        # self._dotenv_values = dotenv_values()

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name.lower())


CONFIG = config()
