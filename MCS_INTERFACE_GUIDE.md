# MCS (Meshcapade Scene) Interface Guide

This document describes the class-based interface for the `smplcodec.mcs` module, which provides a clean, organized way to export MCS files.

## Overview

The interface is built around several key classes that represent different aspects of a meshcapade scene:

- **`CameraIntrinsics`**: Camera intrinsic parameters (focal length, principal point, FOV, etc.)
- **`CameraPose`**: Camera extrinsic parameters (position and orientation)
- **`SMPLCodec`**: SMPL body data
- **`SceneExporter`**: Main exporter class with high-level methods
- **`MCSExporter`**: Base exporter class with low-level functionality

## Quick Start

### Basic Single-Frame Export

```python
from smplcodec.mcs import SceneExporter
from smplcodec.codec import SMPLCodec

# Create exporter
exporter = SceneExporter()

# Load SMPL data using SMPLCodec
body = SMPLCodec.from_file("avatar.smpl")

# Export with default camera (60° FOV, 16:9 aspect ratio)
exporter.export_single_frame([body], "scene.mcs")
```

### Custom Camera Setup

```python
from smplcodec.mcs import CameraIntrinsics, CameraPose
import numpy as np

# Custom camera intrinsics
camera_intrinsics = CameraIntrinsics(
    focal_length=1000.0,
    principal_point=(640.0, 480.0)
)

# Custom camera pose
camera_pose = CameraPose(
    rotation_matrix=np.eye(3, dtype=np.float32),
    translation=np.array([0.0, 0.0, -5.0], dtype=np.float32)
)

# Export with custom camera
exporter.export_single_frame(
    [body], 
    "custom_camera.mcs",
    camera_intrinsics=camera_intrinsics,
    camera_pose=camera_pose
)
```

## Class Reference

### CameraIntrinsics

Represents camera intrinsic parameters.

```python
class CameraIntrinsics:
    def __init__(
        self,
        focal_length: Optional[float] = None,
        principal_point: Optional[Tuple[float, float]] = None,
        yfov_deg: float = 60.0,
        aspect_ratio: float = 16.0 / 9.0,
        znear: float = 0.01
    ):
```

**Parameters:**
- `focal_length`: Camera focal length in pixels
- `principal_point`: Principal point (cx, cy) in pixels
- `yfov_deg`: Vertical field of view in degrees (used when focal_length is None)
- `aspect_ratio`: Image aspect ratio (used when focal_length is None)
- `znear`: Near clipping plane

**Usage:**
```python
# From focal length and principal point
intrinsics = CameraIntrinsics(
    focal_length=1000.0,
    principal_point=(640.0, 480.0)
)

# From FOV and aspect ratio
intrinsics = CameraIntrinsics(
    yfov_deg=45.0,
    aspect_ratio=4.0 / 3.0
)
```

### CameraPose

Represents camera pose (extrinsics).

```python
class CameraPose:
    def __init__(
        self,
        rotation_matrix: NDArray[np.float32],
        translation: NDArray[np.float32]
    ):
```

**Parameters:**
- `rotation_matrix`: 3x3 rotation matrix (must be valid rotation matrix)
- `translation`: 3D translation vector

**Usage:**
```python
import numpy as np

# Identity rotation (no rotation)
rotation = np.eye(3, dtype=np.float32)

# Translation 5 units back
translation = np.array([0.0, 0.0, -5.0], dtype=np.float32)

pose = CameraPose(rotation, translation)
```

### SMPLCodec

Represents an SMPL body with its data. This class is the core SMPL data structure from the `smplcodec.codec` module.

```python
class SMPLCodec:
    # See smplcodec.codec for full class definition
```

**Key Features:**
- Loads SMPL data from files using `SMPLCodec.from_file(filename)`
- Contains pose, shape, and animation data
- Validates SMPL data structure
- Can write to files or buffers

**Usage:**
```python
# Load from file
body = SMPLCodec.from_file("avatar.smpl")

# Create minimal SMPL codec
body = SMPLCodec(
    frame_count=1,
    frame_rate=30.0,
    body_translation=np.array([[0.0, 0.0, 0.0]]),
    body_pose=np.zeros((1, 22, 3))
)
```

### SceneExporter

High-level scene exporter with simplified interface.

#### Methods

##### `export_single_frame()`

Export a single-frame scene.

```python
def export_single_frame(
    self,
    smpl_bodies: List[SMPLCodec],
    output_path: Union[str, Path],
    camera_intrinsics: Optional[CameraIntrinsics] = None,
    camera_pose: Optional[CameraPose] = None
) -> None:
```

**Parameters:**
- `smpl_bodies`: List of SMPLCodec objects
- `output_path`: Output file path
- `camera_intrinsics`: Camera intrinsics (uses defaults if None)
- `camera_pose`: Camera pose (uses identity pose if None)

##### `export_animated_scene()`

Export an animated scene with camera animation.

```python
def export_animated_scene(
    self,
    smpl_bodies: List[SMPLCodec],
    frame_presences: List[List[int]],
    output_path: Union[str, Path],
    num_frames: int,
    frame_rate: float,
    camera_intrinsics: CameraIntrinsics,
    camera_poses: List[CameraPose]
) -> None:
```

**Parameters:**
- `smpl_bodies`: List of SMPLCodec objects
- `frame_presences`: List of frame presence ranges for each body
- `output_path`: Output file path
- `num_frames`: Number of animation frames
- `frame_rate`: Animation frame rate
- `camera_intrinsics`: Camera intrinsics
- `camera_poses`: List of camera poses for each frame

##### `export_static_camera_scene()`

Export a scene with a static camera pose.

```python
def export_static_camera_scene(
    self,
    smpl_bodies: List[SMPLCodec],
    frame_presences: List[List[int]],
    output_path: Union[str, Path],
    num_frames: int,
    frame_rate: float,
    camera_intrinsics: CameraIntrinsics,
    camera_pose: CameraPose,
    static_frame_index: int = 0
) -> None:
```

**Parameters:**
- `smpl_bodies`: List of SMPLCodec objects
- `frame_presences`: List of frame presence ranges for each body
- `output_path`: Output file path
- `num_frames`: Number of animation frames
- `frame_rate`: Animation frame rate
- `camera_intrinsics`: Camera intrinsics
- `camera_pose`: Static camera pose
- `static_frame_index`: Frame index to use for camera pose

## Backward Compatibility

The original function-based interface is still available for backward compatibility:

- `export_single_frame_scene()`
- `export_scene_with_animated_camera()`
- `export_scene_with_static_camera_pose()`

These functions internally use the new class-based system.

## Examples

### Example 1: Multiple SMPL Bodies

```python
# Load multiple SMPL files
bodies = []
frame_presences = []
for i, path in enumerate(["avatar1.smpl", "avatar2.smpl"]):
    body = SMPLCodec.from_file(path)
    bodies.append(body)
    # Each body present in different frame ranges
    frame_presence = [i * 100, (i + 1) * 100]
    frame_presences.append(frame_presence)

# Export scene
exporter.export_single_frame(bodies, "multi_avatar.mcs")
```

### Example 2: Animated Camera

```python
# Create camera poses for animation
camera_poses = []
for frame in range(100):
    # Simple circular motion
    angle = frame * 0.1
    rotation = sp.spatial.transform.Rotation.from_rotvec([0, angle, 0]).as_matrix()
    translation = np.array([np.cos(angle) * 5, 0, np.sin(angle) * 5])
    camera_poses.append(CameraPose(rotation, translation))

# Export animated scene
exporter.export_animated_scene(
    [body],
    "animated_camera.mcs",
    num_frames=100,
    frame_rate=30.0,
    camera_intrinsics=camera_intrinsics,
    camera_poses=camera_poses
)
```

### Example 3: Custom Camera Setup

```python
# Professional camera setup
camera_intrinsics = CameraIntrinsics(
    focal_length=2000.0,  # 50mm equivalent on full frame
    principal_point=(1920.0, 1080.0)  # 4K resolution
)

# Camera positioned for portrait photography
camera_pose = CameraPose(
    rotation_matrix=sp.spatial.transform.Rotation.from_euler(
        'xyz', [np.pi/6, 0, 0]
    ).as_matrix(),
    translation=np.array([0.0, -1.5, 3.0])
)

exporter.export_single_frame(
    [body],
    "portrait_camera.mcs",
    camera_intrinsics=camera_intrinsics,
    camera_pose=camera_pose
)
```

## Error Handling

The new interface includes comprehensive error checking:

- **Invalid rotation matrices**: Automatically validated
- **Type checking**: Proper type hints and validation
- **File I/O errors**: Graceful handling with informative messages

## Performance

The class-based interface is designed for efficiency:

- **Reusable objects**: Create camera setups once, use multiple times
- **Minimal copying**: Efficient data handling
- **Batch operations**: Process multiple bodies efficiently

## Migration from Old Interface

If you're using the old function-based interface, migration is straightforward:

**Old way:**
```python
export_single_frame_scene(
    smpl_buffers,
    "output.mcs",
    use_default_camera=True
)
```

**New way:**
```python
exporter = SceneExporter()
exporter.export_single_frame(smpl_buffers, "output.mcs")
```

The new interface provides the same functionality with better organization and extensibility.
