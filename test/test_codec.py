import numpy as np
import pytest

from pathlib import Path
from smplcodec import SMPLCodec, SMPLGender, SMPLVersion

TESTDIR = Path(__file__).parent / "files"


def test_default():
    s = SMPLCodec()
    assert s.smpl_version == SMPLVersion.SMPLX
    assert s.gender == SMPLGender.NEUTRAL


def test_read_default():
    a = SMPLCodec()
    b = SMPLCodec.from_file(TESTDIR / "default.smpl")
    assert a == b


def test_read_sample():
    a = SMPLCodec.from_file(TESTDIR / "avatar.smpl")
    assert a.smpl_version == SMPLVersion.SMPLX
    assert a.gender == SMPLGender.NEUTRAL

    assert a.shape_parameters.shape == (16,)
    assert a.frame_count == 601
    assert a.frame_rate == 120.0

    assert a.body_pose.shape == (601, 22, 3)
    assert a.body_translation.shape == (601, 3)


def test_full_pose():
    s = SMPLCodec()
    assert s.full_pose.shape == (1, 55, 3)
    a = SMPLCodec.from_file(TESTDIR / "avatar.smpl")
    assert a.full_pose.shape == (601, 55, 3)


def test_single_pose():
    a = SMPLCodec.from_file(TESTDIR / "pose_avatar.smpl")
    assert a.full_pose.shape == (1, 55, 3)
    assert a.body_translation is None


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"smpl_version": SMPLVersion.SMPLH},
        {"smpl_version": 0},
        {"gender": SMPLGender.MALE},
        {"gender": 0},
        {"shape_parameters": np.zeros((10))},
        {"frame_count": 1, "body_pose": np.zeros((1, 22, 3))},
        {"frame_count": 5, "frame_rate": 60.0, "body_pose": np.zeros((5, 22, 3))},
    ],
)
def test_validate_succeeds(params):
    SMPLCodec(**params)


@pytest.mark.parametrize(
    "params",
    [
        {"something_unexpected": None},
        {"smpl_version": None},
        {"smpl_version": -1},
        {"smpl_version": 4},
        {"smpl_gender": -1},
        {"shape_parameters": "something"},
        {"shape_parameters": np.zeros((5, 5))},
        {"body_pose": np.zeros((1, 22, 3))},
        {"frame_count": 5, "body_pose": np.zeros((5, 22, 3))},
        {"frame_count": "many"},
        {"frame_count": 5, "frame_rate": "much"},
        {"frame_count": 5, "frame_rate": 5},
    ],
)
def test_validate_fails(params):
    pytest.raises(TypeError, SMPLCodec, **params)
