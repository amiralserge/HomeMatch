import base64
import numpy as np
from io import BytesIO
from mimetypes import guess_type
from typing import Iterator

from PIL import Image as ImageModule

IMG_SIZE = (640, 427)


def local_image_to_data_url(image_path) -> str:
    """
    Function to encode a local image into data URL
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        # mime_type = 'image/png'
        raise Exception(f"Could not detect mime type of file `{image_path}`")

    img = ImageModule.open(image_path).resize(IMG_SIZE)
    _buffer = BytesIO()
    img.save(_buffer, format=mime_type.split("/")[1])
    base64_encoded_data = base64.b64encode(_buffer.getvalue()).decode("utf-8")
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def split_in_chunks(iterator, chunk_size) -> list[Iterator]:
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
