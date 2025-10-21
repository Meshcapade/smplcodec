import base64
import json
import io
import json
import base64
import logging
import numpy as np
import scipy as sp

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from numpy.typing import NDArray

from .constants import cv_to_gltf_axis_correction
from .codec import SMPLCodec

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class CameraIntrinsics:
    """Represents camera intrinsic parameters."""

    def __init__(
        self,
        focal_length: Optional[float] = None,
        principal_point: Optional[Tuple[float, float]] = None,
        yfov_deg: float = 60.0,
        aspect_ratio: float = 16.0 / 9.0,
        znear: float = 0.01,
    ):
        """
        Initialize camera intrinsics.

        Args:
            focal_length: Camera focal length in pixels
            principal_point: Principal point (cx, cy) in pixels
            yfov_deg: Vertical field of view in degrees (used when focal_length is None)
            aspect_ratio: Image aspect ratio (used when focal_length is None)
            znear: Near clipping plane
        """
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.yfov_deg = yfov_deg
        self.aspect_ratio = aspect_ratio
        self.znear = znear

    def to_gltf_dict(self) -> Dict[str, Any]:
        """Convert to GLTF camera perspective dictionary."""
        if self.focal_length is not None and self.principal_point is not None:
            # Calculate from focal length and principal point
            cx, cy = self.principal_point
            img_width, img_height = cx * 2, cy * 2
            yfov = 2 * np.arctan(img_height / (2 * self.focal_length))
            aspect_ratio = img_width / img_height
        else:
            # Use default values
            yfov = np.deg2rad(self.yfov_deg)
            aspect_ratio = self.aspect_ratio

        return {
            "yfov": float(yfov),
            "znear": float(self.znear),
            "aspectRatio": float(aspect_ratio),
        }


class CameraPose:
    """Represents camera pose (extrinsics)."""

    def __init__(self, rotation_matrix: NDArray[np.float32], translation: NDArray[np.float32]):
        """
        Initialize camera pose.

        Args:
            rotation_matrix: 3x3 rotation matrix
            translation: 3D translation vector
        """
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)
        self.translation = np.asarray(translation, dtype=np.float32)

        # Validate rotation matrix
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must be 3x3")
        if self.translation.shape != (3,):
            raise ValueError("translation must be 3D vector")

        # Check if rotation matrix is valid
        if not np.allclose(self.rotation_matrix.T @ self.rotation_matrix, np.eye(3), atol=1e-3):
            raise ValueError("rotation_matrix is not a valid rotation matrix")

    def to_gltf_pose(self) -> Tuple[List[float], List[float]]:
        """Convert to GLTF translation and rotation (quaternion)."""

        # Camera position in world coords: C = -R^T * t
        cam_pos = -self.rotation_matrix.T @ self.translation

        # Node rotation (world): R_world = R^T * cv_to_gltf
        R_gltf = self.rotation_matrix.T @ cv_to_gltf_axis_correction
        quat_xyzw = sp.spatial.transform.Rotation.from_matrix(R_gltf).as_quat()

        return (
            [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
            [float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])],
        )


class MCSExporter:
    """Main class for exporting MCS (Meshcapade Scene) files."""

    def __init__(self):
        """Initialize the MCS exporter."""
        pass

    def _create_base_gltf(self, num_frames: int) -> Dict[str, Any]:
        """Create the base GLTF structure."""
        return {
            "asset": {"version": "2.0", "generator": "SMPLCodec MCS Exporter"},
            "scene": 0,
            "scenes": [
                {"nodes": [0], "extensions": {"MC_scene_description": {"num_frames": num_frames, "smpl_bodies": []}}}
            ],
            "nodes": [
                {"name": "RootNode", "children": [1]},
                {
                    "name": "AnimatedCamera",
                    "camera": 0,
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
            ],
            "cameras": [{"type": "perspective", "perspective": {}}],
            "buffers": [],
            "bufferViews": [],
            "accessors": [],
            "animations": [],
            "extensionsUsed": ["MC_scene_description"],
        }

    def _add_smpl_bodies(self, gltf: Dict[str, Any], bodies: List[SMPLCodec], frame_presences: List[List[int]]) -> None:
        """Add SMPL bodies to the GLTF structure."""
        for i, (body, frame_presence) in enumerate(zip(bodies, frame_presences)):
            # Get binary data from SMPLCodec
            body_data = self._get_smpl_binary_data(body)

            # Add elf.uffer
            gltf["buffers"].append(
                {
                    "byteLength": len(body_data),
                    "uri": f"data:application/octet-stream;base64,{base64.b64encode(body_data).decode('utf-8')}",
                }
            )

            # Add buffer view
            gltf["bufferViews"].append({"buffer": i, "byteOffset": 0, "byteLength": len(body_data)})

            # Add to scene description
            gltf["scenes"][0]["extensions"]["MC_scene_description"]["smpl_bodies"].append(
                {"frame_presence": frame_presence, "bufferView": i}
            )

    def _get_smpl_binary_data(self, smpl_codec: SMPLCodec) -> bytes:
        """Get binary data from SMPLCodec object by writing to a temporary buffer."""
        import io

        buffer = io.BytesIO()
        smpl_codec.write_to_buffer(buffer)
        return buffer.getvalue()

    def _add_camera_animation(
        self, gltf: Dict[str, Any], poses: List[CameraPose], num_frames: int, frame_rate: float
    ) -> None:
        """Add camera animation to the GLTF structure."""
        times = np.arange(num_frames, dtype=np.float32) * (1.0 / frame_rate)

        camera_positions = np.zeros((num_frames, 3), dtype=np.float32)
        rotations = np.zeros((num_frames, 4), dtype=np.float32)

        for i, pose in enumerate(poses):
            if i >= num_frames:
                break

            translation, rotation = pose.to_gltf_pose()
            camera_positions[i] = translation
            rotations[i] = rotation

        # Add buffers
        buffers_start_idx = len(gltf["buffers"])
        gltf["buffers"].extend(
            [
                {
                    "byteLength": times.nbytes,
                    "uri": f"data:application/octet-stream;base64,{base64.b64encode(times.tobytes()).decode('utf-8')}",
                },
                {
                    "byteLength": camera_positions.nbytes,
                    "uri": f"data:application/octet-stream;base64,{base64.b64encode(camera_positions.tobytes()).decode('utf-8')}",
                },
                {
                    "byteLength": rotations.nbytes,
                    "uri": f"data:application/octet-stream;base64,{base64.b64encode(rotations.tobytes()).decode('utf-8')}",
                },
            ]
        )

        # Add buffer views
        gltf["bufferViews"].extend(
            [
                {"name": "TimeBufferView", "buffer": buffers_start_idx, "byteOffset": 0, "byteLength": times.nbytes},
                {
                    "name": "camera_track_translations_buffer_view",
                    "buffer": buffers_start_idx + 1,
                    "byteOffset": 0,
                    "byteLength": camera_positions.nbytes,
                },
                {
                    "name": "camera_track_rotations_buffer_view",
                    "buffer": buffers_start_idx + 2,
                    "byteOffset": 0,
                    "byteLength": rotations.nbytes,
                },
            ]
        )

        # Add accessors
        gltf["accessors"].extend(
            [
                {
                    "name": "TimeAccessor",
                    "bufferView": len(gltf["bufferViews"]) - 3,
                    "componentType": 5126,
                    "count": num_frames,
                    "type": "SCALAR",
                    "min": [float(times.min())],
                    "max": [float(times.max())],
                },
                {
                    "name": "camera_track_translations_accessor",
                    "bufferView": len(gltf["bufferViews"]) - 2,
                    "componentType": 5126,
                    "count": num_frames,
                    "type": "VEC3",
                },
                {
                    "name": "camera_track_rotations_accessor",
                    "bufferView": len(gltf["bufferViews"]) - 1,
                    "componentType": 5126,
                    "count": num_frames,
                    "type": "VEC4",
                },
            ]
        )

        # Add animation
        gltf["animations"].append(
            {
                "channels": [
                    {"sampler": 0, "target": {"node": 1, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
                ],
                "samplers": [
                    {
                        "input": len(gltf["accessors"]) - 3,
                        "interpolation": "LINEAR",
                        "output": len(gltf["accessors"]) - 2,
                    },
                    {
                        "input": len(gltf["accessors"]) - 3,
                        "interpolation": "LINEAR",
                        "output": len(gltf["accessors"]) - 1,
                    },
                ],
            }
        )

    def _write_gltf(self, gltf: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write GLTF data to file."""
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(gltf, f, indent=2)


class SceneExporter(MCSExporter):
    """High-level scene exporter with simplified interface."""

    def export_single_frame(
        self,
        smpl_bodies: Union[List[SMPLCodec], List[bytes]],
        output_path: Union[str, Path],
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        camera_pose: Optional[CameraPose] = None,
    ) -> None:
        """
        Export a single-frame scene.

        Args:
            smpl_bodies: List of SMPLCodec objects or raw bytes data
            output_path: Output file path
            camera_intrinsics: Camera intrinsics (uses defaults if None)
            camera_pose: Camera pose (uses identity pose if None)
        """
        # Convert bytes to SMPLCodec if needed
        bodies = []
        frame_presences = []
        for body in smpl_bodies:
            if isinstance(body, bytes):
                # For bytes, we need to create a temporary SMPLCodec or handle differently
                # For now, we'll require SMPLCodec objects
                raise ValueError("Raw bytes are no longer supported. Please use SMPLCodec objects.")
            else:
                bodies.append(body)
                frame_presences.append([0, 1])  # Single frame presence

        # Create GLTF structure
        gltf = self._create_base_gltf(num_frames=1)

        # Set camera intrinsics
        if camera_intrinsics is None:
            camera_intrinsics = CameraIntrinsics()
        gltf["cameras"][0]["perspective"] = camera_intrinsics.to_gltf_dict()

        # Add SMPL bodies
        self._add_smpl_bodies(gltf, bodies, frame_presences)

        # Set camera pose
        if camera_pose is not None:
            translation, rotation = camera_pose.to_gltf_pose()
            gltf["nodes"][1]["translation"] = translation
            gltf["nodes"][1]["rotation"] = rotation

        # Ensure no animations
        gltf["animations"] = []

        # Write file
        self._write_gltf(gltf, output_path)
        log.info(f"Single-frame scene exported to {output_path}")

    def export_animated_scene(
        self,
        smpl_bodies: List[SMPLCodec],
        frame_presences: List[List[int]],
        output_path: Union[str, Path],
        num_frames: int,
        frame_rate: float,
        camera_intrinsics: CameraIntrinsics,
        camera_poses: List[CameraPose],
    ) -> None:
        """
        Export an animated scene with camera animation.

        Args:
            smpl_bodies: List of SMPLCodec objects
            frame_presences: List of frame presence ranges for each body
            output_path: Output file path
            num_frames: Number of animation frames
            frame_rate: Animation frame rate
            camera_intrinsics: Camera intrinsics
            camera_poses: List of camera poses for each frame
        """
        if len(camera_poses) != num_frames:
            raise ValueError(f"Expected {num_frames} camera poses, got {len(camera_poses)}")
        if len(smpl_bodies) != len(frame_presences):
            raise ValueError(f"Expected {len(smpl_bodies)} frame_presences, got {len(frame_presences)}")

        # Create GLTF structure
        gltf = self._create_base_gltf(num_frames)

        # Set camera intrinsics
        gltf["cameras"][0]["perspective"] = camera_intrinsics.to_gltf_dict()

        # Add SMPL bodies
        self._add_smpl_bodies(gltf, smpl_bodies, frame_presences)

        # Add camera animation
        self._add_camera_animation(gltf, camera_poses, num_frames, frame_rate)

        # Write file
        self._write_gltf(gltf, output_path)
        log.info(f"Animated scene exported to {output_path}")

    def export_static_camera_scene(
        self,
        smpl_bodies: List[SMPLCodec],
        frame_presences: List[List[int]],
        output_path: Union[str, Path],
        num_frames: int,
        frame_rate: float,
        camera_intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        static_frame_index: int = 0,
    ) -> None:
        """
        Export a scene with a static camera pose.

        Args:
            smpl_bodies: List of SMPLCodec objects
            frame_presences: List of frame presence ranges for each body
            output_path: Output file path
            num_frames: Number of animation frames
            frame_rate: Animation frame rate
            camera_intrinsics: Camera intrinsics
            camera_pose: Static camera pose
            static_frame_index: Frame index to use for camera pose
        """
        if len(smpl_bodies) != len(frame_presences):
            raise ValueError(f"Expected {len(smpl_bodies)} frame_presences, got {len(frame_presences)}")

        # Create GLTF structure
        gltf = self._create_base_gltf(num_frames)

        # Set camera intrinsics
        gltf["cameras"][0]["perspective"] = camera_intrinsics.to_gltf_dict()

        # Add SMPL bodies
        self._add_smpl_bodies(gltf, smpl_bodies, frame_presences)

        # Set static camera pose
        translation, rotation = camera_pose.to_gltf_pose()
        gltf["nodes"][1]["translation"] = translation
        gltf["nodes"][1]["rotation"] = rotation

        # No animations
        gltf["animations"] = []

        # Write file
        self._write_gltf(gltf, output_path)
        log.info(f"Static camera scene exported to {output_path}")


def extract_smpl_from_mcs(mcs_file_path: Union[str, Path]) -> List[SMPLCodec]:
    """
    Extract SMPL body parameters from an MCS file.

    Args:
        mcs_file_path: Path to the .mcs file

    Returns:
        List of SMPLCodec objects, one per body in the scene
    """

    # Load the GLTF data from the MCS file
    with open(mcs_file_path, "r", encoding="utf-8") as f:
        gltf_data = json.load(f)

    # Extract SMPL bodies from the GLTF structure
    scene_ext = gltf_data["scenes"][0]["extensions"]["MC_scene_description"]
    smpl_bodies = scene_ext["smpl_bodies"]

    log.debug(f"Found {len(smpl_bodies)} SMPL bodies in the MCS file")

    extracted_bodies = []
    for i, smpl_body in enumerate(smpl_bodies):
        buffer_view_idx = smpl_body["bufferView"]
        buffer_view = gltf_data["bufferViews"][buffer_view_idx]
        buffer_idx = buffer_view["buffer"]
        buffer_data = gltf_data["buffers"][buffer_idx]

        uri = buffer_data["uri"]
        if uri.startswith("data:application/octet-stream;base64,"):
            base64_data = uri.split(",", 1)[1]
            smpl_buffer = base64.b64decode(base64_data)

            # Convert bytes to SMPLCodec object
            buffer_io = io.BytesIO(smpl_buffer)
            smpl_codec = SMPLCodec.from_file(buffer_io)  # type: ignore[arg-type]
            extracted_bodies.append(smpl_codec)

            log.debug(f"Extracted SMPL body {i}: {len(smpl_buffer)} bytes")
        else:
            raise ValueError(f"Unexpected URI format in buffer {buffer_idx}: {uri}")

    return extracted_bodies


# Convenience functions for backward compatibility and easy usage
def export_single_frame_scene(
    smpl_buffers: Union[List[bytes], List[SMPLCodec]],
    output_path: Union[str, Path],
    camera_intrinsics: Optional[CameraIntrinsics] = None,
    camera_pose: Optional[CameraPose] = None,
) -> None:
    """
    Export a single-frame scene (convenience function).

    Args:
        smpl_buffers: List of SMPLCodec objects or raw bytes data
        output_path: Output file path
        camera_intrinsics: Camera intrinsics (uses defaults if None)
        camera_pose: Camera pose (uses identity pose if None)
    """
    exporter = SceneExporter()
    exporter.export_single_frame(smpl_buffers, output_path, camera_intrinsics, camera_pose)


if __name__ == "__main__":
    # Example usage with new class-based interface
    exporter = SceneExporter()

    # Load SMPL data using SMPLCodec
    try:
        smpl_path = "test/files/avatar.smpl"
        body = SMPLCodec.from_file(smpl_path)

        # Export with default camera
        exporter.export_single_frame([body], "example_default_camera.mcs")

        # Export with custom camera
        camera_intrinsics = CameraIntrinsics(focal_length=1000.0, principal_point=(640.0, 480.0))
        camera_pose = CameraPose(
            rotation_matrix=np.eye(3, dtype=np.float32), translation=np.array([0.0, 0.0, -5.0], dtype=np.float32)
        )

        exporter.export_single_frame([body], "example_custom_camera.mcs", camera_intrinsics, camera_pose)

    except FileNotFoundError:
        log.error("Example SMPL file not found. Please ensure test/files/avatar.smpl exists")
