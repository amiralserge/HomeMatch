import functools


def singleton(init_once: bool = False):
    """
    Decorator for creating singleton classes.

    :param init_once: If True, __init__ is called only once.
                      If False, __init__ is called on every instance creation.
    """

    def inner(klass):
        original__init__ = klass.__init__
        original__new__ = klass.__new__
        klass._instance = None
        klass._instance_initialized = False

        @functools.wraps(original__init__)
        def __init__(self, *args, **kwargs):
            if init_once and self.__class__._instance_initialized:
                return
            original__init__(self, *args, **kwargs)
            self.__class__._instance_initialized = True

        @functools.wraps(original__new__)
        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super(klass, cls).__new__(cls, *args, **kwargs)
            return cls._instance

        klass.__new__ = __new__
        klass.__init__ = __init__
        return klass

    return inner
