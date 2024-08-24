from unittest import mock

import pytest

from config import config


class TestConfig:

    @pytest.fixture
    @classmethod
    def setup(cls):
        config._instance_initialized = False
        yield
        config._instance_initialized = False

    def test_singleton(self, setup):

        cfg1 = config()
        cfg2 = config()

        assert cfg1 is cfg2

        cfg2.llm_model = "some_model"
        cfg1.llm_temperature = 0

        assert cfg1.llm_model == cfg2.llm_model
        assert cfg1.llm_temperature == cfg2.llm_temperature
        assert cfg1.llm_model == cfg2.llm_model == "some_model"
        assert cfg1.llm_temperature == cfg2.llm_temperature == 0

    @mock.patch("config.dotenv_values")
    def test_env(self, dotenv_values_mock, setup):

        dotenv_values_mock.return_value = {
            "LLM_MODEL": "gpt-4o-mini",
            "LLM_TEMPERATURE": 1,
        }
        cfg1 = config()
        cfg2 = config()
        assert cfg1 is cfg2

        assert cfg1.llm_model == cfg2.LLM_MODEL == "gpt-4o-mini"
        assert cfg1.LLM_TEMPERATURE == cfg2.llm_temperature == 1
