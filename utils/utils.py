import base64
from io import BytesIO
from mimetypes import guess_type
from typing import List, Tuple

import numpy as np
from PIL import Image as ImageModule
from PIL.Image import Image

IMG_SIZE = (640, 427)


def open_image(image_path) -> Image:
    return ImageModule.open(image_path)


def resize_image(image: Image, size=IMG_SIZE) -> Image:
    return image.resize(size)


def b64encode_image(image: Image, format: str) -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def local_image_to_data_url(image_path, size: Tuple = IMG_SIZE) -> str:
    """
    Function to encode a local image into data URL
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        # mime_type = 'image/png'
        raise Exception(f"Could not detect mime type of file `{image_path}`")
    format = mime_type.split("/")[-1]

    image = open_image(image_path)
    if size:
        image = resize_image(image, size=size)
    # Construct the data URL
    return f"data:{mime_type};base64,{b64encode_image(image, format=format)}"


def split_in_chunks(iterator: List, chunk_size: int) -> list[List]:
    if not len(iterator):
        return []
    indices = np.arange(chunk_size, len(iterator), chunk_size)
    return list(map(lambda i: i.tolist(), np.array_split(iterator, indices)))


class singleton:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance
