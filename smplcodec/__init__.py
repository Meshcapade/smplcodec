"A simple codec class for SMPL data."
from .codec import SMPLCodec, SMPLGender, SMPLVersion, SMPLParamStructure
from .version import __version__

__all__ = [SMPLCodec, SMPLGender, SMPLVersion, SMPLParamStructure, __version__]
