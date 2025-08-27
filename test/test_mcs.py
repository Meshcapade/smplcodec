# test_exporters.py
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scipy.spatial.transform import Rotation as R_scipy

from smplcodec.mcs import (
    SceneExporter,
    CameraIntrinsics,
    CameraPose,
)
from smplcodec.codec import SMPLCodec



def _read_gltf(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dummy_smpl_codec(n=1):
    """Create dummy SMPLCodec objects for testing."""
    codecs = []
    for i in range(n):
        # Create minimal SMPLCodec with dummy data
        codec = SMPLCodec(
            frame_count=1,
            frame_rate=30.0,
            body_translation=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            body_pose=np.zeros((1, 22, 3), dtype=np.float32)
        )
        codecs.append(codec)
    return codecs


# -------- single-frame (default camera) --------


def test_single_frame_default_camera_intrinsics_and_pose(tmp_path: Path):
    out = tmp_path / "single_default.mcs"
    smpl_codecs = _dummy_smpl_codec(2)

    exporter = SceneExporter()
    exporter.export_single_frame(
        smpl_bodies=smpl_codecs,
        output_path=str(out)
    )

    gltf = _read_gltf(out)

    # Structure + metadata
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["asset"]["generator"] == "SMPLCodec MCS Exporter"
    assert gltf["scenes"][0]["extensions"]["MC_scene_description"]["num_frames"] == 1
    assert len(gltf["animations"]) == 0

    # SMPL buffers + presence
    smpl_meta = gltf["scenes"][0]["extensions"]["MC_scene_description"]["smpl_bodies"]
    assert len(smpl_meta) == 2
    assert all(m["frame_presence"] == [0, 1] for m in smpl_meta)
    assert len(gltf["buffers"]) == 2
    assert len(gltf["bufferViews"]) == 2

    # Camera intrinsics (default)
    persp = gltf["cameras"][0]["perspective"]
    assert math.isclose(persp["yfov"], math.radians(60.0), rel_tol=1e-6)
    assert math.isclose(persp["aspectRatio"], 16 / 9, rel_tol=1e-6)
    assert math.isclose(persp["znear"], 0.01, rel_tol=1e-9)

    # Default pose: origin + identity
    cam_node = gltf["nodes"][1]
    assert cam_node["translation"] == [0.0, 0.0, 0.0]
    assert cam_node["rotation"] == [0.0, 0.0, 0.0, 1.0]


# -------- single-frame (custom camera) --------


def test_single_frame_with_custom_camera(tmp_path: Path):
    out = tmp_path / "single_custom.mcs"
    smpl_codecs = _dummy_smpl_codec(1)

    # Create custom camera setup
    camera_intrinsics = CameraIntrinsics(
        focal_length=800.0,
        principal_point=(500.0, 400.0)
    )

    camera_pose = CameraPose(
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.zeros((3,), dtype=np.float32)
    )

    exporter = SceneExporter()
    exporter.export_single_frame(
        smpl_bodies=smpl_codecs,
        output_path=str(out),
        camera_intrinsics=camera_intrinsics,
        camera_pose=camera_pose
    )

    gltf = _read_gltf(out)

    assert gltf["scenes"][0]["extensions"]["MC_scene_description"]["num_frames"] == 1
    assert len(gltf["animations"]) == 0

    # Intrinsics from focal length and principal point
    persp = gltf["cameras"][0]["perspective"]
    expected_aspect = 500.0 / 400.0
    expected_yfov = 2 * math.atan(400.0 / 800.0)
    assert math.isclose(persp["aspectRatio"], expected_aspect, rel_tol=1e-6)
    assert math.isclose(persp["yfov"], expected_yfov, rel_tol=1e-6)

    # Pose: should be at origin with identity rotation
    cam_node = gltf["nodes"][1]
    assert cam_node["translation"] == [0.0, 0.0, 0.0]
    # Rotation should be identity (no rotation) - note: quaternion format may vary
    # The important thing is that it represents no rotation
    quat = np.array(cam_node["rotation"])
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0]) or np.allclose(quat, [0.0, 0.0, 0.0, 1.0])


def test_single_frame_custom_camera_intrinsics_validation():
    """Test that CameraIntrinsics validates inputs correctly."""

    # Valid inputs
    intrinsics = CameraIntrinsics(
        focal_length=1000.0,
        principal_point=(640.0, 480.0)
    )
    assert intrinsics.focal_length == 1000.0
    assert intrinsics.principal_point == (640.0, 480.0)

    # Default values
    intrinsics = CameraIntrinsics()
    assert intrinsics.focal_length is None
    assert intrinsics.principal_point is None
    assert intrinsics.yfov_deg == 60.0
    assert intrinsics.aspect_ratio == 16.0 / 9.0


def test_camera_pose_validation():
    """Test that CameraPose validates inputs correctly."""

    # Valid inputs
    pose = CameraPose(
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.array([0.0, 0.0, -5.0], dtype=np.float32)
    )
    assert pose.rotation_matrix.shape == (3, 3)
    assert pose.translation.shape == (3,)

    # Invalid rotation matrix
    with pytest.raises(ValueError, match="rotation_matrix must be 3x3"):
        CameraPose(
            rotation_matrix=np.eye(2, dtype=np.float32),
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )

    # Invalid translation
    with pytest.raises(ValueError, match="translation must be 3D vector"):
        CameraPose(
            rotation_matrix=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0], dtype=np.float32)
        )


# -------- multi-frame: static camera pose --------


def test_static_camera_scene_multi_frame(tmp_path: Path):
    out = tmp_path / "static_pose_multi.mcs"
    smpl_codecs = _dummy_smpl_codec(1)

    num_frames = 5
    frame_presences = [[0, num_frames - 1]]

    # Create camera poses for multiple frames
    camera_poses = []
    for i in range(num_frames):
        pose = CameraPose(
            rotation_matrix=np.eye(3, dtype=np.float32),
            translation=np.array([i, 2 * i, -i], dtype=np.float32)
        )
        camera_poses.append(pose)

    # Use frame 3 for static pose
    static_pose = camera_poses[3]

    camera_intrinsics = CameraIntrinsics(
        focal_length=800.0,
        principal_point=(500.0, 500.0)
    )

    exporter = SceneExporter()
    exporter.export_static_camera_scene(
        smpl_bodies=smpl_codecs,
        frame_presences=frame_presences,
        output_path=str(out),
        num_frames=num_frames,
        frame_rate=30.0,
        camera_intrinsics=camera_intrinsics,
        camera_pose=static_pose,
        static_frame_index=0  # Not used in this method
    )

    gltf = _read_gltf(out)

    # No animations for static pose
    assert len(gltf["animations"]) == 0

    # Camera should be positioned at the static pose
    cam_node = gltf["nodes"][1]
    # Translation should be transformed from CV to GLTF coordinates
    # The transformation is: cam_pos = -R^T * t
    # With R = identity and t = [3, 6, -3], we get [-3, -6, 3]
    assert cam_node["translation"] == [-3.0, -6.0, 3.0]


# -------- multi-frame: animated camera --------


def test_animated_camera_scene(tmp_path: Path):
    out = tmp_path / "animated_camera.mcs"
    smpl_codecs = _dummy_smpl_codec(1)

    num_frames = 3
    frame_rate = 30.0
    frame_presences = [[0, num_frames - 1]]

    # Create camera poses for animation
    camera_poses = []
    for i in range(num_frames):
        # Keep rotations identity, translate along Z
        pose = CameraPose(
            rotation_matrix=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, float(i)], dtype=np.float32)
        )
        camera_poses.append(pose)

    camera_intrinsics = CameraIntrinsics(
        focal_length=800.0,
        principal_point=(500.0, 500.0)
    )

    exporter = SceneExporter()
    exporter.export_animated_scene(
        smpl_bodies=smpl_codecs,
        frame_presences=frame_presences,
        output_path=str(out),
        num_frames=num_frames,
        frame_rate=frame_rate,
        camera_intrinsics=camera_intrinsics,
        camera_poses=camera_poses
    )

    gltf = _read_gltf(out)

    # Animation structure
    assert len(gltf["animations"]) == 1
    anim = gltf["animations"][0]
    assert len(anim["channels"]) == 2  # translation + rotation
    assert len(anim["samplers"]) == 2

    # Time accessor min/max
    time_accessor = next(a for a in gltf["accessors"] if a["name"] == "TimeAccessor")
    assert time_accessor["count"] == num_frames
    assert math.isclose(time_accessor["min"][0], 0.0, abs_tol=1e-8)
    expected_max_t = (num_frames - 1) / frame_rate
    assert math.isclose(time_accessor["max"][0], expected_max_t, rel_tol=1e-6)

    # Ensure buffers/views were appended for time/pos/rot
    # We expect: initial SMPL buffers + 3 extra buffers from animation
    # 1 SMPL buffer + 3 animation buffers = 4 total
    assert len(gltf["buffers"]) == 1 + 3
    # 1 SMPL view + 3 animation views = 4 total
    assert len(gltf["bufferViews"]) == 1 + 3


# -------- error handling --------


def test_animated_scene_mismatched_camera_poses(tmp_path: Path):
    """Test that animated scene validates camera pose count."""
    out = tmp_path / "mismatched_poses.mcs"
    smpl_codecs = _dummy_smpl_codec(1)

    num_frames = 5
    frame_presences = [[0, num_frames - 1]]

    # Only 3 poses for 5 frames
    camera_poses = []
    for i in range(3):
        pose = CameraPose(
            rotation_matrix=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, float(i)], dtype=np.float32)
        )
        camera_poses.append(pose)

    camera_intrinsics = CameraIntrinsics()

    exporter = SceneExporter()

    with pytest.raises(ValueError, match="Expected 5 camera poses, got 3"):
        exporter.export_animated_scene(
            smpl_bodies=smpl_codecs,
            frame_presences=frame_presences,
            output_path=str(out),
            num_frames=num_frames,
            frame_rate=30.0,
            camera_intrinsics=camera_intrinsics,
            camera_poses=camera_poses
        )


def test_static_camera_scene_mismatched_frame_presences(tmp_path: Path):
    """Test that static camera scene validates frame presence count."""
    out = tmp_path / "mismatched_presences.mcs"
    smpl_codecs = _dummy_smpl_codec(2)  # 2 bodies

    num_frames = 5
    frame_presences = [[0, 4]]  # Only 1 presence for 2 bodies

    camera_pose = CameraPose(
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.array([0.0, 0.0, -5.0], dtype=np.float32)
    )

    camera_intrinsics = CameraIntrinsics()

    exporter = SceneExporter()

    with pytest.raises(ValueError, match="Expected 2 frame_presences, got 1"):
        exporter.export_static_camera_scene(
            smpl_bodies=smpl_codecs,
            frame_presences=frame_presences,
            output_path=str(out),
            num_frames=num_frames,
            frame_rate=30.0,
            camera_intrinsics=camera_intrinsics,
            camera_pose=camera_pose
        )
