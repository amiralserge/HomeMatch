import base64
import io
from io import BytesIO
from mimetypes import guess_type
from typing import Tuple

from PIL import Image as ImageModule
from PIL.Image import Image

IMG_SIZE = (640, 427)


def open_image(image_path) -> Image:
    return ImageModule.open(image_path)


def pil_to_bytes(image: Image, format: str = "jpeg") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def b64encode_image(image: Image, format: str) -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def resize_image(image: Image, size=IMG_SIZE) -> Image:
    return image.resize(size)


def local_image_to_data_url(image_path, size: Tuple = IMG_SIZE) -> str:
    """
    Function to encode a local image into data URL
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        # mime_type = 'image/png'
        raise Exception(f"Could not detect mime type of file `{image_path}`")
    img_format = mime_type.split("/")[-1]

    image = open_image(image_path)
    if size:
        image = resize_image(image, size=size)
    # Construct the data URL
    return f"data:{mime_type};base64,{b64encode_image(image, format=img_format)}"
