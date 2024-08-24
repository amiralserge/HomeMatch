from . import register_app_mode


def run():
    raise NotImplementedError()


register_app_mode(app_fn=run, name="form")
