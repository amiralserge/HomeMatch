from typing import Any

from dotenv import dotenv_values, load_dotenv

from utils import singleton

load_dotenv()


@singleton
class config(object):
    def __init__(self) -> None:
        for key, value in dotenv_values().items():
            setattr(self, key.lower(), value)
        # self._dotenv_values = dotenv_values()

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name.lower())


CONFIG = config()
