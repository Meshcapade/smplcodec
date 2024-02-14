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
