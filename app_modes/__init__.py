# flake8: noqa
import functools
import logging

_logger = logging.getLogger(__name__)
APP_MODES = {}


class UnknownAppModeException(Exception):
    pass


def log(app_fn):
    @functools.wraps(app_fn)
    def inner(*args, **kwargs):
        logging.basicConfig(
            format="{asctime} HomeMatch {levelname}: {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
            level=logging.INFO,
        )
        app_fn(*args, **kwargs)

    return inner


def register_app_mode(app_fn: callable, name="") -> None:
    APP_MODES[name] = log(app_fn)


def run_app(mode) -> None:
    if app_fn := APP_MODES.get(mode):
        app_fn()
    else:
        raise UnknownAppModeException(f"Unknown app mode '{mode}'")


from . import chat, form
