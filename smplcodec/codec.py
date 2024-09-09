import logging
import numpy as np

from contextlib import closing
from dataclasses import dataclass, asdict, fields
from enum import IntEnum
from typing import Optional
from numpy.typing import ArrayLike

from .utils import extract_item, coerce_type, matching, to_camel, to_snake
from .version import MAJOR


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SMPLVersion(IntEnum):
    SMPL = 0
    SMPLH = 1
    SMPLX = 2
    SUPR = 3
    SMPLPP = 4
    SKEL = 5

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f'invalid model name {value}')


class SMPLGender(IntEnum):
    NEUTRAL = 0
    MALE = 1
    FEMALE = 2

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f'invalid gender {value}')


@dataclass
class PoseParameterSizes:
    body_pose: tuple = None
    head_pose: Optional[tuple] = None
    left_hand_pose: Optional[tuple] = None
    right_hand_pose: Optional[tuple] = None


SMPLParamStructure = {
    # all tuples
    SMPLVersion.SMPL: PoseParameterSizes(body_pose=(24, 3)),
    SMPLVersion.SMPLH: PoseParameterSizes(body_pose=(22, 3), left_hand_pose=(15, 3), right_hand_pose=(15, 3)),
    SMPLVersion.SMPLX: PoseParameterSizes(
        body_pose=(22, 3), head_pose=(3, 3), left_hand_pose=(15, 3), right_hand_pose=(15, 3)
    ),
    SMPLVersion.SUPR: PoseParameterSizes(
        body_pose=(22, 3),
        head_pose=(3, 3),
        left_hand_pose=(15, 3),
        right_hand_pose=(15, 3),  # ,
        # left_foot_pose=(10, 3),
        # righ_foot_pose=(10,3),
    ),
    SMPLVersion.SMPLPP: PoseParameterSizes(body_pose=(46,)),
    SMPLVersion.SKEL: PoseParameterSizes(body_pose=(46,)),
}


@dataclass
class SMPLCodec:
    codec_version: int = MAJOR
    smpl_version: SMPLVersion = SMPLVersion.SMPLX
    gender: SMPLGender = SMPLGender.NEUTRAL

    shape_parameters: Optional[ArrayLike] = None  # [10-300] betas
    expression_parameters: Optional[ArrayLike] = None  # [10-100] FLAME parameters

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
        for field in fields(SMPLParamStructure[self.smpl_version]):
            part_pose = getattr(self, field.name)
            if part_pose is None:
                part_pose = np.zeros(
                    ((count,) + getattr(SMPLParamStructure[self.smpl_version], field.name))
                )  # merge tuples for shape
            pose = np.append(pose, part_pose, axis=1)
        return pose

    def __post_init__(self):
        self.validate()

    def __eq__(self, other):
        return all(matching(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))

    @classmethod
    def from_file(cls, filename):
        with closing(np.load(filename)) as infile:
            data = {to_snake(k): extract_item(v) for (k, v) in dict(infile).items()}
            if "codec_version" not in data:
                data["codec_version"] = 1
        return cls(**data)

    @classmethod
    def from_amass_npz(cls, filename: str, smpl_version_str: str = "smplx"):
        with closing(np.load(filename, allow_pickle=True)) as infile:
            in_dict = dict(infile)

            mapped_dict = {
                "shape_parameters": in_dict.get("betas", None),
                "body_translation": in_dict.get("trans", None),
                "gender": SMPLGender.from_string(str(in_dict.get("gender", "neutral"))),
                "smpl_version": SMPLVersion.from_string(str(in_dict.get("surface_model_type", smpl_version_str))),
                "frame_count": int(
                    in_dict.get("frameCount", in_dict["trans"].shape[0])
                ),  # expects trans to be not None if frameCount does not exist
                # for frame rate, different names seem to exist in AMASS (SMPLH: mocap_framerate, SMPLX: mocap_frame_rate?!)
                "frame_rate": float(
                    in_dict.get("frameRate", in_dict.get("mocap_frame_rate", in_dict.get("mocap_framerate", 60.0)))
                ),  # default 60.0??
            }

            # check if pose parameters are stored separately
            if "body_pose" in in_dict:
                mapped_dict["body_pose"] = np.concatenate(
                    (in_dict["root_orient"], in_dict["pose_body"]), axis=-1
                ).reshape(mapped_dict["frame_count"], -1, 3)

                if "pose_jaw" in in_dict and "pose_eye" in in_dict:
                    mapped_dict["head_pose"] = np.concatenate(
                        (in_dict["pose_jaw"], in_dict["pose_eye"]), axis=-1
                    ).reshape(mapped_dict["frame_count"], -1, 3)

                if "pose_hand" in in_dict:
                    mapped_dict["left_hand_pose"] = in_dict["pose_hand"][
                        :, : int(in_dict["pose_hand"].shape[-1] / 2.0)
                    ].reshape(mapped_dict["frame_count"], -1, 3)
                    mapped_dict["right_hand_pose"] = in_dict["pose_hand"][
                        :, int(in_dict["pose_hand"].shape[-1] / 2.0) :
                    ].reshape(mapped_dict["frame_count"], -1, 3)
            elif "poses" in in_dict:
                # split "full pose" into separate parameters
                joint_info = SMPLParamStructure[mapped_dict["smpl_version"]]
                start_ind = 0

                for field in fields(joint_info):
                    num_params = np.prod(getattr(joint_info, field.name))

                    if num_params is not None:
                        mapped_dict[field.name] = in_dict["poses"][:, start_ind : start_ind + num_params].reshape(
                            (-1,) + getattr(joint_info, field.name)
                        )
                        start_ind += num_params
            else:
                print("No pose parameters in file!!!")

            return cls(**mapped_dict)

    def write(self, filename):
        self.validate()
        data = {to_camel(f): coerce_type(v) for f, v in asdict(self).items() if v is not None}
        with open(filename, "wb") as outfile:
            np.savez_compressed(outfile, **data)

    def validate(self):
        try:
            self.smpl_version = SMPLVersion(self.smpl_version)
            self.gender = SMPLGender(self.gender)

            if self.shape_parameters is not None:
                assert len(self.shape_parameters.shape) == 1, "Bad shape_parameters"

            if self.frame_count is not None:
                assert isinstance(self.frame_count, int), "frame_count should be int"
                if self.frame_count > 1:
                    assert isinstance(self.frame_rate, float), "frame_rate should be float"

                for attr, shape in [("body_translation", (self.frame_count, 3))] + [
                    (field.name, ((self.frame_count,) + getattr(SMPLParamStructure[self.smpl_version], field.name)))
                    for field in fields(SMPLParamStructure[self.smpl_version])
                    if getattr(SMPLParamStructure[self.smpl_version], field.name) is not None
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
