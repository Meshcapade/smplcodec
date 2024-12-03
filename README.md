# SMPL Codec

SMPLCodec is a minimal pure-Python library that provides a standardized way to read and write SMPL(-H/-X) parameters of bodies and animations as `.smpl` files.

See the [SMPL Wiki](https://meshcapade.wiki/SMPL) for a general description of the model, and the [SMPL-X](https://smpl-x.is.tue.mpg.de/) project page and [GitHub](https://github.com/vchoutas/smplx) for model data and code for the most commonly used version.


## Installation

```bash
pip install smplcodec
```

## Usage

A `.smpl` files is simply an NPZ that follows some conventions. It is flat-structured and only contains numeric (specifically `int32` and `float32`) data for maximum interoperability. The library provides a `SMPLCodec` dataclass which provides convenience methods for reading, writing, and validating SMPL data.

```
    from smplcodec import SMPLCodec, SMPLVersion, SMPLGender

    # Read a 601-frame sequence from a file
    a = SMPLCodec.from_file("avatar.smpl")

    # The full_pose helper property contains the sequence data
    assert a.full_pose.shape == (601, 55, 3)            
    
    # You can also load AMASS sequences
    a = SMPLCodec.from_amass_npz("motion.npz")

    # Create a new neutral avatar and write it to file
    b = SMPLCodec(smpl_version=SMPLVersion.SMPLX, gender=SMPLGender.NEUTRAL, shape_parameters=np.zeros(10))
    b.write("neutral.smpl")
```

## License

This library is released under the MIT license.

