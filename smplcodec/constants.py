import numpy as np

# Convert from CV to GLTF convention
cv_to_gltf_axis_correction = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],  # Flip Y (down to up)
        [0.0, 0.0, -1.0],  # Flip Z (forward to backward)
    ],
    dtype=np.float32,
)
