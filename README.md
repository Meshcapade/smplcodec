# SMPL Codec

SMPLCodec is a minimal pure-Python library that provides a standardized way to read and write SMPL(-H/-X) parameters of bodies and animations as `.smpl` files, and export meshcapde scenes as `.mcs` files.

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

## Meshcapade Scene (MCS) Export

SMPLCodec also provides functionality to export meshcapade scenes as `.mcs` files (GLTF format with SMPL extensions). The class-based interface makes it easy to create scenes with custom cameras and SMPL bodies:

```python
from smplcodec.mcs import SceneExporter, CameraIntrinsics, CameraPose
from smplcodec.codec import SMPLCodec
import numpy as np

# Create exporter
exporter = SceneExporter()

# Load SMPL data using SMPLCodec
body = SMPLCodec.from_file("avatar.smpl")

# Custom camera setup
camera_intrinsics = CameraIntrinsics(focal_length=1000.0, principal_point=(640.0, 480.0))
camera_pose = CameraPose(
    rotation_matrix=np.eye(3, dtype=np.float32),
    translation=np.array([0.0, 0.0, -5.0], dtype=np.float32)
)

# Export scene
exporter.export_single_frame([body], "scene.mcs", camera_intrinsics, camera_pose)
```

## Extracting SMPL Bodies from MCS Files

You can extract SMPL bodies from MCS files and use them in the [Meshcapade Editor](https://me.meshcapade.com/editor):

```python
from smplcodec.mcs import extract_smpl_from_mcs

# Extract all SMPL bodies from MCS file
bodies = extract_smpl_from_mcs("scene.mcs")
print(f"Found {len(bodies)} SMPL bodies")

# Save as individual .smpl files
for i, body in enumerate(bodies):
    body.write(f"extracted_bodies/smpl_body_{i}.smpl")
```

The extracted `.smpl` files can be drag-and-dropped directly into the [Meshcapade Editor](https://me.meshcapade.com/editor) to visualize and edit the bodies.

https://github.com/Meshcapade/smplcodec/assets/drag_smpl_to_editor.mp4

For detailed documentation, see [MCS_INTERFACE_GUIDE.md](MCS_INTERFACE_GUIDE.md).

## License

This library is released under the MIT license.
