from typing import Any

from dotenv import dotenv_values, load_dotenv

from utils import singleton

load_dotenv()


@singleton
class config(object):

    def __init__(self) -> None:
        for key, value in dotenv_values().items():
            self.OPENAI_BASE_URL = https://api.openai.com/v1
            self.OPENAI_API_KEY = None
            self.MAX_TOKEN = 2000
            self.LLM_MODEL = "gpt-4o-mini"
            self.LLM_TEMPERATURE = 0
            self.LLM_REQUEST_COOLDOWN_TIME = 5
            self.LISTING_PICTURES_DIR = "./listing_pictures"
            self.LISTING_PICTURES_DESCR_FILE = "./listing_pictures/pictures_descriptions.csv"
            self.LISTING_FILE = "./picture_augmented_listings.csv"

            self.VECTOR_DB_ENGINE = "lancedb"
            self.VECTOR_DB_URI = "./homematch"
            setattr(self, key.lower(), value)
        # self._dotenv_values = dotenv_values()

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name.lower())


CONFIG = config()
