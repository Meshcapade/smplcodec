import logging
import numpy as np

from contextlib import closing
from dataclasses import dataclass, asdict, fields
from enum import IntEnum
from typing import Optional
from numpy.typing import ArrayLike

from .utils import extract_item, coerce_type, matching, to_camel, to_snake


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


JOINTS = {
    SMPLVersion.SMPLX: {
        "body_pose": 22,
        "head_pose": 3,
        "left_hand_pose": 15,
        "right_hand_pose": 15,
    }
}


@dataclass
class SMPLCodec:
    smpl_version: SMPLVersion = SMPLVersion.SMPLX
    gender: SMPLGender = SMPLGender.NEUTRAL

    shape_parameters: Optional[ArrayLike] = None  # [10-300] betas

    # motion metadata
    frame_count: Optional[int] = None
    frame_rate: Optional[float] = None

    # pose / motion data
    body_translation: Optional[ArrayLike] = None  # [N x 3] Global trans
    body_pose: Optional[ArrayLike] = None  # [N x 22 x 3] pelvis..right_wrist
    head_pose: Optional[ArrayLike] = None  # [N x 3 x 3] jaw, leftEye, rightEye
    left_hand_pose: Optional[ArrayLike] = None  # [N x 15 x 3] left_index1..left_thumb3
    right_hand_pose: Optional[ArrayLike] = None  # [N x 15 x 3] right_index1..right_thumb3

    @property
    def full_pose(self) -> ArrayLike:
        """Create and return the full_pose [frame_count x num_joints x 3].
        If frame_count is 0 or None it is assumed to be 1 instead.
        This function will always return a full pose array, if any pose
        information is missing it will be filled with zeros automatic.
        """
        count = self.frame_count or 1
        pose = np.empty((count, 0, 3))
        for part, num_joints in JOINTS[self.smpl_version].items():
            part_pose = getattr(self, part)
            if part_pose is None:
                part_pose = np.zeros((count, num_joints, 3))
            pose = np.append(pose, part_pose, axis=1)
        return pose

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

            if self.shape_parameters is not None:
                assert len(self.shape_parameters.shape) == 1, "Bad shape_parameters"

            if self.frame_count is not None:
                assert isinstance(self.frame_count, int), "frame_count should be int"
                if self.frame_count > 1:
                    assert isinstance(self.frame_rate, float), "frame_rate should be float"

                for attr, shape in [("body_translation", (self.frame_count, 3))] + [
                    (key, (self.frame_count, num, 3)) for key, num in JOINTS[self.smpl_version].items()
                ]:
                    value = getattr(self, attr)
                    if value is not None:
                        assert getattr(self, attr).shape == shape, f"{attr} shape should be {shape}"
            else:
                for attr in (
                    "body_translation",
                    "body_pose",
                    "head_pose",
                    "left_hand_pose",
                    "right_hand_pose",
                ):
                    assert getattr(self, attr) is None, f"{attr} exists but no frame_count"

        except (AttributeError, ValueError, AssertionError) as e:
            raise TypeError(f"Failed to validate SMPL Codec object: {e}") from e
