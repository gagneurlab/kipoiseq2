from itertools import islice
from typing import Iterable, TypeVar

import numpy as np


def to_scalar(obj):
    """Convert numpy scalar to native scalar"""
    if isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def parse_alphabet(alphabet):
    if isinstance(alphabet, str):
        return list(alphabet)
    else:
        return alphabet


def parse_dtype(dtype):
    if isinstance(dtype, str):
        try:
            return eval(dtype)
        except Exception as e:
            raise ValueError("Unable to parse dtype: {}. \nException: {}".format(dtype, e))
    else:
        return dtype


T = TypeVar("T")


def batch_iter(items: Iterable[T], batch_size: int) -> Iterable[Iterable[T]]:
    # ensure this is an iterator
    item_iter = iter(items)
    while True:
        # create next `batch_size` number of items;
        batch = list(islice(item_iter, batch_size))
        if len(batch) == 0:
            break

        yield batch
