from typing import List

import numpy as np


def split_in_chunks(iterator: List, chunk_size: int) -> list[List]:
    if not len(iterator):
        return []
    indices = np.arange(chunk_size, len(iterator), chunk_size)
    return list(map(lambda i: i.tolist(), np.array_split(iterator, indices)))
