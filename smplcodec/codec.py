import logging
import numpy as np

from contextlib import closing
from dataclasses import dataclass, asdict, fields
from enum import IntEnum
from typing import Optional
from numpy.typing import ArrayLike


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SMPLVersion(IntEnum):
    SMPL = 0
    SMPLH = 1
    SMPLX = 2
    SUPR = 3


class SMPLGender(IntEnum):
    NEUTRAL = 0
    MALE = 1
    FEMALE = 2


@dataclass
class SMPLCodec:
    smpl_version: SMPLVersion = SMPLVersion.SMPLX
    gender: SMPLGender = SMPLGender.NEUTRAL

    shape_parameters: Optional[ArrayLike] = None

    # motion metadata
    frame_count: Optional[int] = None
    frame_rate: Optional[float] = None

    # optional time-slice
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    # pose / motion data
    body_translation: Optional[ArrayLike] = None    # [N x 3] Global trans
    body_pose: Optional[ArrayLike] = None           # [N x 22 x 3] pelvis..right_wrist
    head_pose: Optional[ArrayLike] = None           # [N x 3 x 3] jaw, leftEye, rightEye
    left_hand_pose: Optional[ArrayLike] = None       # [N x 15 x 3] left_index1..left_thumb3
    right_hand_pose: Optional[ArrayLike] = None      # [N x 15 x 3] right_index1..right_thumb3

    def __post_init__(self):
        self.validate()

    def __eq__(self, other):
        return all(matching(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))

    @classmethod
    def from_file(cls, filename):
        with closing(np.load(filename)) as infile:
            return cls(**{to_snake(k): extract_item(v) for (k, v) in dict(infile).items()})

    def write(self, filename):
        data = {to_camel(f): coerce_type(v) for f, v in asdict(self).items() if v is not None}
        with open(filename, "wb") as outfile:
            np.savez_compressed(outfile, **data)

    def validate(self):
        try:
            self.smplVersion = SMPLVersion(self.smpl_version)
            self.gender = SMPLGender(self.gender)

            assert self.shape_parameters is None or len(self.shape_parameters.shape) == 1

            if self.frame_count is not None:
                assert isinstance(self.frame_count, int)
                if self.frame_count > 1:
                    assert isinstance(self.frame_rate, float)

            assert self.start_seconds is None or isinstance(self.start_seconds, float)
            assert self.end_seconds is None or isinstance(self.end_seconds, float)

            assert self.body_translation is None or self.body_translation.shape == (self.frame_count, 3)
            assert self.body_pose is None or self.body_pose.shape == (self.frame_count, 22, 3)
            assert self.head_pose is None or self.head_pose.shape == (self.frame_count, 3, 3)
            assert self.left_hand_pose is None or self.left_hand_pose.shape == (self.frame_count, 15, 3)
            assert self.right_hand_pose is None or self.right_hand_pose.shape == (self.frame_count, 15, 3)

        except (ValueError, AssertionError) as e:
            raise TypeError("Failed to validate SMPL Codec object") from e


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
