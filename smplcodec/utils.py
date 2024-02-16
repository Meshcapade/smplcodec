import numpy as np


def extract_item(thing):
    if thing is None:
        return None
    if thing.shape == ():
        return thing.item()
    return thing


def coerce_type(thing):
    if thing is None:
        return None
    if isinstance(thing, int):
        return np.array(thing, dtype=np.int32)
    if isinstance(thing, float):
        return np.array(thing, dtype=np.float32)
    if isinstance(thing, np.ndarray):
        # All non-scalar fields contain float data
        return thing.astype(np.float32, casting="same_kind")
    raise ValueError(f"Wrong kind of thing: {thing}")


def matching(thing, other):
    if thing is None:
        return other is None
    if isinstance(thing, int) or isinstance(thing, float):
        return thing == other
    if isinstance(thing, np.ndarray) and isinstance(other, np.ndarray):
        # All non-scalar fields contain float data
        return np.allclose(thing, other)
    return False


def to_snake(name):
    return "".join(c if c.islower() else f"_{c.lower()}" for c in name).lstrip("_")


def to_camel(name):
    first, *rest = name.split("_")
    return first + "".join(word.capitalize() for word in rest)
