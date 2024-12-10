import numpy as np
import os

from typing import Optional, Union
from numpy.typing import ArrayLike

PathType = Union[str, bytes, os.PathLike]


def extract_item(thing: Optional[np.ndarray]) -> Union[None, np.ndarray]:
    if thing is None:
        return None
    if thing.shape == ():
        return thing.item()
    return thing


def coerce_type(thing: Union[int, float, ArrayLike]) -> ArrayLike:
    if isinstance(thing, int):
        return np.array(thing, dtype=np.int32)
    if isinstance(thing, float):
        return np.array(thing, dtype=np.float32)
    if isinstance(thing, np.ndarray):
        # All non-scalar fields contain float data
        return thing.astype(np.float32, casting="same_kind")
    raise ValueError(f"Wrong kind of thing: {thing}")


def matching(thing, other) -> bool:
    if thing is None:
        return other is None
    if isinstance(thing, int) or isinstance(thing, float):
        return thing == other
    if isinstance(thing, np.ndarray) and isinstance(other, np.ndarray):
        # All non-scalar fields contain float data
        return np.allclose(thing, other)
    return False


def to_snake(name: str) -> str:
    return "".join(c if c.islower() else f"_{c.lower()}" for c in name).lstrip("_")


def to_camel(name: str) -> str:
    first, *rest = name.split("_")
    return first + "".join(word.capitalize() for word in rest)
